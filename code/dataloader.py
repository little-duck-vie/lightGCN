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
    def __init__(self, path="../data/lastfm", n_clusters=5, svd_dim=50):
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
        build_cluster_sampler_state(self, n_clusters=n_clusters, svd_dim=svd_dim, seed=42, verbose=True)
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()

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

    def __init__(self,config = world.config,path="../data/gowalla", n_clusters=world.config['cluster_num'], svd_dim=world.config['svd_dim']):
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
        build_cluster_sampler_state(self, n_clusters=n_clusters, svd_dim=svd_dim, seed=42, verbose=True)
        self.__testDict = self.__build_test()
#-----------------------------------------------------------------------------------------------------------
#############--------------------Newly Negative Sample for cluster sampler---------------------#############
#-----------------------------------------------------------------------------------------------------------

        print("Building extended positive items via neighbor sharing...")

        # 1. Build user_pos and item_users
        self.user_pos = {u: set(self._allPos[u]) for u in range(self.n_user)}
        self.item_users = {i: set() for i in range(self.m_item)}
        for u, items in self.user_pos.items():
            for i in items:
                self.item_users[i].add(u)

        # 2. Calculate item frequency and remove popular items
        item_freq = {i: len(users) for i, users in self.item_users.items()}
        hub_threshold = 200  # or adjust via config
        self.popular_items = {i for i, c in item_freq.items() if c > hub_threshold}
        print(f"Filtered out {len(self.popular_items)} popular items (>{hub_threshold} users).")

        # 3. Compute neighbors and extended positives
        import heapq
        M = 5  # top-M neighbors, can be moved to args
        self.user_neighbors = {}
        self.extended_pos_items = {}

        for u in range(self.n_user):
            counter = {}
            for i in self.user_pos[u]:
                if i in self.popular_items:
                    continue
                for v in self.item_users[i]:
                    if v == u:
                        continue
                    counter[v] = counter.get(v, 0) + 1
            top_neighbors = heapq.nlargest(M, counter.items(), key=lambda x: x[1])
            self.user_neighbors[u] = [v for v, _ in top_neighbors]

            # Build extended item set from neighbors
            extended_candidate_items = []

            for v in self.user_neighbors[u]:
                # Lấy các item của hàng xóm v, bỏ item đã có của u
                for i in self.user_pos[v]:
                    if i not in self.user_pos[u]:
                        extended_candidate_items.append(i)

            # Đếm độ phổ biến trong các hàng xóm
            from collections import Counter
            item_counts = Counter(extended_candidate_items)

            # Sắp xếp theo phổ biến nhất (trong các hàng xóm)
            sorted_items = [item for item, count in item_counts.most_common()]

            # Giới hạn số lượng: k = 1/2 số lượng pos gốc
            k = max(1, len(self.user_pos[u]))
            top_k_items = sorted_items[:k]

            self.extended_pos_items[u] = set(top_k_items)
        print(f"pos user 0 original: {len(self.user_pos[0])}, extended: {len(self.extended_pos_items[0])}")
        self.__testDict = self.__build_test()
        self._build_neg_sampling_helpers()
        # build_allPosNew(self)
        print(f"{world.dataset} is ready to go")
            
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

    def _build_neg_sampling_helpers(self):
        # 1) posSet để check O(1)
        self.posSet = [set(p) for p in self.allPos]   # hoặc self._allPos, tùy code bạn dùng

        # 2) item -> users CSR
        self.item_user_csr = self.UserItemNet.T.tocsr()

        # 3) popularity (giảm bias)
        self.item_pop = np.asarray(self.UserItemNet.sum(axis=0)).ravel().astype(np.int64)

#-----------------------------------------------------------------------------------------------------------
#############--------------------Newly Negative Sample for cluster sampler---------------------#############
#-----------------------------------------------------------------------------------------------------------





def build_cluster_sampler_state(dataset, n_clusters, svd_dim=64, seed=42, verbose=True):
    """
    Precompute state cho cluster-aware negative sampling (RAM nhẹ).
    dataset cần có:
      - dataset.UserItemNet (CSR UxI)
      - dataset._allPos
      - dataset.n_users, dataset.m_items
    Tạo ra:
      - dataset.item_cluster
      - dataset.cluster2items
      - dataset.posSet
      - dataset.user_dom_cluster
    """
    if verbose:
        print(f"[ClusterNeg] build state: n_clusters={n_clusters}, svd_dim={svd_dim}")

    # (I, U)
    X = dataset.UserItemNet.T.tocsr()
    U = X.shape[1]
    if U <= 1:
        svd_dim_eff = 1
    else:
        svd_dim_eff = min(svd_dim, U - 1)

    svd = TruncatedSVD(n_components=svd_dim_eff, random_state=seed)
    item_emb = svd.fit_transform(X)  # (I, svd_dim)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=2048,
        n_init="auto"
    )
    item_cluster = kmeans.fit_predict(item_emb).astype(np.int32)

    dataset.item_cluster = item_cluster
    dataset.cluster2items = [np.where(item_cluster == c)[0].astype(np.int32) for c in range(n_clusters)]

    # positives set để membership O(1)
    dataset.posSet = [set(p) for p in dataset._allPos]

    # dominant cluster của mỗi user
    dataset.user_dom_cluster = np.zeros(dataset.n_users, dtype=np.int32)
    for u in range(dataset.n_users):
        pos = dataset._allPos[u]
        if len(pos) == 0:
            dataset.user_dom_cluster[u] = 0
            continue
        cs = item_cluster[np.array(pos, dtype=np.int32)]
        dataset.user_dom_cluster[u] = int(np.bincount(cs, minlength=n_clusters).argmax())

    if verbose:
        print("[ClusterNeg] done.")
# def build_allPosNew(dataset, verbose=True):
#     """
#     allPosNew[u] = positives cũ + tất cả item thuộc dominant cluster của user u
#     Lưu dưới dạng set để membership O(1)
#     """
#     if verbose:
#         print("[AllPosNew] building...")

#     # pos cũ
#     posSet = [set(p) for p in dataset._allPos]

#     # allPosNew
#     allPosNew = [None] * dataset.n_users
#     for u in range(dataset.n_users):
#         dom = int(dataset.user_dom_cluster[u])
#         dom_items = dataset.cluster2items[dom]  # numpy array item ids
#         # union: pos cũ + toàn bộ item trong dominant cluster
#         s = posSet[u].copy()
#         s.update(dom_items.tolist())
#         if hasattr(dataset, "extended_pos_items") and dataset.extended_pos_items[u]:
#             s.update(dataset.extended_pos_items[u])
#         allPosNew[u] = s

#     dataset.posSet = posSet                 # vẫn giữ nếu bạn cần
#     dataset.allPosNew = allPosNew           # NEW
#     print(f"all pos user 0{len(posSet[0])}")
#     print(f"all pos new user 0 {len(allPosNew[0])}")
#     if verbose:
#         print("[AllPosNew] done.")

def sample_same_cluster_negative(dataset, user, positem, max_trials=50):
    """
    positem thuộc cluster c nào => negitem cũng lấy trong cluster c đó.
    Reject nếu neg nằm trong positive của user.
    """
    c = int(dataset.item_cluster[positem])
    items = dataset.cluster2items[c]
    pos = dataset.posSet[user]
    extended_pos = dataset.extended_pos_items[user] 
    if len(items) == 0:
        # fallback: random toàn bộ items
        while True:
            neg = int(np.random.randint(0, dataset.m_items))
            if neg not in pos and neg not in extended_pos:
                return neg

    # thử lấy trong cùng cluster
    for _ in range(max_trials):
        neg = int(items[np.random.randint(0, len(items))])
        if neg not in pos and neg not in extended_pos:
            return neg

    # fallback nếu cluster bị "đầy" positives đối với user
    while True:
        neg = int(np.random.randint(0, dataset.m_items))
        if neg not in pos:
            return neg

def sample_negative_allPosNew(dataset, user, max_trials=200):
    """
    Random negative sampling với điều kiện:
      neg NOT IN allPosNew[user]
    """
    ban = dataset.allPosNew[user]

    for _ in range(max_trials):
        neg = int(np.random.randint(0, dataset.m_items))
        if neg not in ban:
            return neg

    # fallback (chắc chắn ra)
    while True:
        neg = int(np.random.randint(0, dataset.m_items))
        if neg not in ban:
            return neg



#-----------------------------------------------------------------------------------------------------------
#############--------------------Newly Negative Sample for cluster sampler---------------------#############
#-----------------------------------------------------------------------------------------------------------


