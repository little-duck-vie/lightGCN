"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """
    def __init__(self, path="../data/lastfm", n_clusters=10, svd_dim=50):
        # train or test
        cprint("loading [last fm]")
        self.mode_dict = {'train':0, "test":1}
        self.mode    = self.mode_dict['train']
        # self.n_users = 1892
        # self.m_items = 4489
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        # print(trainData.head())
        testData  = pd.read_table(join(path, 'test1.txt'), header=None)
        # print(testData.head())
        trustNet  = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        # print(trustNet[:5])
        trustNet -= 1
        trainData-= 1
        testData -= 1
        self.trustNet  = trustNet
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        # self.trainDataSize = len(self.trainUser)
        self.testUser  = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = np.array(testData[:][1])
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")
        
        # (users,users)
        self.socialNet    = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_users,self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items)) 
        
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()

#-----------------------------------------------------------------------------------------------------------
#############--------------------Newly Negative Sample for cluster sampler---------------------#############
#-----------------------------------------------------------------------------------------------------------

        item_user_matrix = self.UserItemNet.T   # sparse matrix

        # Giảm chiều bằng TruncatedSVD (phù hợp với sparse matrix)
        svd = TruncatedSVD(n_components=svd_dim, random_state=42)
        item_factors = svd.fit_transform(item_user_matrix)

        # Phân cụm item
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=256
        )
        item_cluster_labels = kmeans.fit_predict(item_factors)
        cluster_to_items = {}
        for item_id, c in enumerate(item_cluster_labels):
            cluster_to_items.setdefault(c, set()).add(item_id)
        self.hardNeg = []
        self.easyNeg = []

        all_items_set = set(range(self.m_items))

        for u in range(self.n_users):
            pos_items = set(self._allPos[u])

            # User không có positive (trường hợp hiếm)
            if len(pos_items) == 0:
                self.hardNeg.append(np.array([], dtype=np.int64))
                self.easyNeg.append(np.array(list(all_items_set), dtype=np.int64))
                continue

            # Đếm xem positive của user thuộc cụm nào nhiều nhất
            pos_clusters = [item_cluster_labels[i] for i in pos_items]
            cluster_count = np.bincount(pos_clusters, minlength=n_clusters)
            dominant_cluster = cluster_count.argmax()

            # Hard negative: cùng cụm dominant nhưng không phải positive
            items_in_dom = cluster_to_items[dominant_cluster]
            hard_neg = items_in_dom - pos_items

            # Easy negative: ngoài cụm dominant và không phải positive
            easy_neg = (all_items_set - items_in_dom) - pos_items

            self.hardNeg.append(np.array(list(hard_neg), dtype=np.int64))
            self.easyNeg.append(np.array(list(easy_neg), dtype=np.int64))

#-----------------------------------------------------------------------------------------------------------
#############--------------------Newly Negative Sample for cluster sampler---------------------#############
#-----------------------------------------------------------------------------------------------------------

    @property
    def n_users(self):
        return 1892
    
    @property
    def m_items(self):
        return 4489
    
    @property
    def trainDataSize(self):
        return len(self.trainUser)
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)
            
            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim+self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D==0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense/D_sqrt
            dense = dense/D_sqrt.t()
            index = dense.nonzero()
            data  = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems

    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user
    
    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']
    
    def __len__(self):
        return len(self.trainUniqueUsers)

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self,config = world.config,path="../data/gowalla", n_clusters=10, svd_dim=50):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
#-----------------------------------------------------------------------------------------------------------
#############--------------------Newly Negative Sample for cluster sampler---------------------#############
#-----------------------------------------------------------------------------------------------------------

        item_user_matrix = self.UserItemNet.T   # sparse matrix

        # Giảm chiều bằng TruncatedSVD (phù hợp với sparse matrix)
        svd = TruncatedSVD(n_components=svd_dim, random_state=42)
        item_factors = svd.fit_transform(item_user_matrix)

        # Phân cụm item
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=256
        )
        item_cluster_labels = kmeans.fit_predict(item_factors)
        cluster_to_items = {}
        for item_id, c in enumerate(item_cluster_labels):
            cluster_to_items.setdefault(c, set()).add(item_id)
        self.hardNeg = []
        self.easyNeg = []

        all_items_set = set(range(self.m_items))

        for u in range(self.n_users):
            pos_items = set(self._allPos[u])

            # User không có positive (trường hợp hiếm)
            if len(pos_items) == 0:
                self.hardNeg.append(np.array([], dtype=np.int64))
                self.easyNeg.append(np.array(list(all_items_set), dtype=np.int64))
                continue

            # Đếm xem positive của user thuộc cụm nào nhiều nhất
            pos_clusters = [item_cluster_labels[i] for i in pos_items]
            cluster_count = np.bincount(pos_clusters, minlength=n_clusters)
            dominant_cluster = cluster_count.argmax()

            # Hard negative: cùng cụm dominant nhưng không phải positive
            items_in_dom = cluster_to_items[dominant_cluster]
            hard_neg = items_in_dom - pos_items

            # Easy negative: ngoài cụm dominant và không phải positive
            easy_neg = (all_items_set - items_in_dom) - pos_items

            self.hardNeg.append(np.array(list(hard_neg), dtype=np.int64))
            self.easyNeg.append(np.array(list(easy_neg), dtype=np.int64))
            
#-----------------------------------------------------------------------------------------------------------
#############--------------------Newly Negative Sample for cluster sampler---------------------#############
#-----------------------------------------------------------------------------------------------------------
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems
    def build_cluster_sampler_state(dataset, n_clusters=100, svd_dim=64, seed=42, verbose=True):
        """
        RAM-light cluster state:
        - dataset.item_cluster: (m_items,)
        - dataset.cluster2items: list[np.array]
        - dataset.user_dom_cluster: (n_users,)
        - dataset.posSet: list[set]  (tối ưu check positive)
        Yêu cầu dataset có:
        - dataset.UserItemNet (CSR, shape UxI)
        - dataset._allPos
        - dataset.n_users, dataset.m_items
        """
        if verbose:
            print(f"[ClusterNeg] building clusters: n_clusters={n_clusters}, svd_dim={svd_dim}")

        # Item-user sparse matrix: (I, U)
        X = dataset.UserItemNet.T.tocsr()

        # SVD dim phải <= U-1
        U = X.shape[1]
        if U <= 1:
            svd_dim_eff = 1
        else:
            svd_dim_eff = min(svd_dim, U - 1)

        # TruncatedSVD trên sparse
        svd = TruncatedSVD(n_components=svd_dim_eff, random_state=seed)
        item_emb = svd.fit_transform(X)  # (I, svd_dim)

        # MiniBatchKMeans để nhanh hơn kmeans thường
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            batch_size=2048,
            n_init="auto"
        )
        item_cluster = kmeans.fit_predict(item_emb).astype(np.int32)  # (I,)

        dataset.item_cluster = item_cluster

        # cluster2items: list các item id thuộc cluster c
        dataset.cluster2items = []
        for c in range(n_clusters):
            dataset.cluster2items.append(np.where(item_cluster == c)[0].astype(np.int32))

        # posSet để membership O(1) (rất quan trọng về tốc độ)
        dataset.posSet = [set(p) for p in dataset._allPos]

        # dominant cluster cho mỗi user
        dataset.user_dom_cluster = np.zeros(dataset.n_users, dtype=np.int32)
        for u in range(dataset.n_users):
            pos = dataset._allPos[u]
            if len(pos) == 0:
                dataset.user_dom_cluster[u] = 0
                continue
            cs = item_cluster[np.array(pos, dtype=np.int32)]
            cnt = np.bincount(cs, minlength=n_clusters)
            dataset.user_dom_cluster[u] = int(cnt.argmax())

        if verbose:
            print("[ClusterNeg] done.")
def sample_cluster_negative(dataset, user, p_hard=0.3, max_trials=50):
    """
    Sample 1 negative item cho user:
      - hard (p_hard): lấy item từ dominant cluster của user
      - easy (1-p_hard): lấy item từ cluster khác dominant
    Reject nếu item nằm trong positives của user.
    """
    dom = int(dataset.user_dom_cluster[user])
    pos = dataset.posSet[user]
    n_clusters = len(dataset.cluster2items)

    use_hard = (np.random.random() < p_hard)

    if use_hard:
        clusters = [dom]
    else:
        # chọn cluster khác dom
        # (tránh tạo list dài mỗi lần, ta random cho nhanh)
        clusters = None

    for _ in range(max_trials):
        if use_hard:
            c = dom
        else:
            # random cluster != dom
            c = np.random.randint(0, n_clusters - 1)
            if c >= dom:
                c += 1

        items = dataset.cluster2items[c]
        if len(items) == 0:
            continue
        neg = int(items[np.random.randint(0, len(items))])
        if neg not in pos:
            return neg

    # fallback: random toàn cục (ít khi xảy ra)
    while True:
        neg = np.random.randint(0, dataset.m_items)
        if neg not in pos:
            return int(neg)