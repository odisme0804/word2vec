import numpy as np
from numpy.linalg import norm
import scipy.stats as st
from collections import defaultdict
from math import *
import progressbar
import time
import multiprocessing as mpc

def get_default_hparas():
    class hparas():
        pass
    hparas.successive_time_constrain = 6
    hparas.precision = 20
    hparas.top_N = 10
    hparas.embedding_dim = 300
    hparas.learn_rate = 0.005
    hparas.max_iter = 10
    hparas.epsilon = 0.0001
    hparas.window_size = 3
    hparas.seq_length = 3
    hparas.tree_type = "huffman"
    hparas.model = "cbow"
    hparas.decay = 1
    hparas.inverse = 1
    # for simularity
    hparas.simu_metric = None # [min, max, avg]
    hparas.simu_func = None # [cos, jaccard]

    return hparas

class GeneralTreeNode():
    def __init__(self, value, freq, group=[]):
        # common part of leaf node and tree node
        self.freq = freq
        self.left = None  # 
        self.right = None # 
        # value of leaf node  will be the word, and be
        # mid vector in tree node
        self.value = value # the value of word 
        self.path = "" # store the path or huffman code
        self.word_list = group

    def __str__(self):
        return 'TreeNode object, value: {v}, freq: {f}, path: {p}'\
            .format(v=self.value,f=self.freq,p=self.path)

class GeneralTree():
    def __init__(self, hparas=None, word_dict=None,
                 word_list=None, user_list=None, user_item_matrix=None):
        self.root = None
        if hparas is None:
            self.hparas = get_default_hparas()
        else:
            self.hparas = hparas
        # for simularity
        self.dist_matrix = None
        self.ui_vector_dict = None
        self.resutls = None
        """ 
        word_dict : each element is a dict with key 'word', including: freq, vector, path(code)
        user_item_matrix : Matrix['User']['LOC'] = '1/0' for Simulartiy Tree.  Like:  defaultdict(lambda: defaultdict(int))
        """
 
        if self.hparas.tree_type == "huffman":
            node_list = [GeneralTreeNode(key, value['freq']) for key, value in word_dict.items()]
            self.build_huffman_tree(node_list)
            self.generate_huffman_code(self.root, word_dict)
        elif self.hparas.tree_type == "simularity":
            self.cal_distance_matrix(word_list, user_list, user_item_matrix)
            node_list = [GeneralTreeNode(key, value['freq'], [key]) for key, value in word_dict.items()]
            self.build_simularity_tree(node_list)
            self.generate_huffman_code(self.root, word_dict)

    def cal_distance_matrix(self, word_list, user_list, user_item_matrix):
        # initial user-item-matrix
        ui_vector_dict = {}
        for word in word_list:
            ui_vector_dict[word] = np.zeros(shape=[1, len(user_list)])
        
        # mark user who visited
        for idx, user in enumerate(user_item_matrix):
            for word, count in user_item_matrix[user].items():
                if count > 0:
                    ui_vector_dict[word][0][idx] = 1
        self.ui_vector_dict = ui_vector_dict

        cnttt = 0
        allll = 0
        for word in word_list:
            if np.all(ui_vector_dict[word] == 0):
                cnttt += 1
            allll += 1
        print(cnttt, allll)

        tmpp = 0
        # calculate distance between words
        print("compute distance metrix")
        # ver1 
        """ 
        distbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(),progressbar.Bar()],maxval=len(word_list)).start()
        self.dist_matrix = defaultdict(lambda: defaultdict(float))
        for i, word1 in enumerate(word_list):
            time.sleep(0.01)
            distbar.update(i)
            print(i)
            for j, word2 in enumerate(word_list):
                 if i == j: 
                     self.dist_matrix[word1][word2] = self.cal_sim(ui_vector_dict[word1],
                                                                   ui_vector_dict[word2])
                 elif i < j:
                     self.dist_matrix[word1][word2] = self.cal_sim(ui_vector_dict[word1],
                                                               ui_vector_dict[word2])
                     self.dist_matrix[word2][word1] = self.dist_matrix[word1][word2]

                 if self.dist_matrix[word2][word1] == 0.0:
                     tmpp += 1
        distbar.finish()
        print(tmpp)
        self.display_dist_matrix()
        """
        # ver2
        self.dist_matrix = defaultdict(lambda: defaultdict(float))
        #tep = self.hparas
        #self.hparas = None
        
        manager = mpc.Manager()
        return_dict = manager.dict()

        #pool = mpc.Pool()
        distbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(),progressbar.Bar()],maxval=len(word_list)).start()
        jobs = []
        for word in word_list:
            #result = pool.apply_async(self.word_simus, args=(word, word_list))

            if len(jobs) >= 10:
                for i,x in enumerate(jobs):
                    distbar.update(i+1)
                    x.join()

            p = mpc.Process(target=self.word_simus, args=(word, word_list, self.hparas.simu_func, return_dict))
            jobs.append(p)
            p.start()


        for i,x in enumerate(jobs):
            distbar.update(i+1)
            x.join()

        #pool.close()
        #pool.join()
        
        """       
        for result, word1 in zip(results, word_list):
            val = result.get()
            for i, word2 in enumerate(word_list):
                #self.dist_matrix[word1][word2] = val[i]
                print(val[i])"""
        for result, word1 in zip(return_dict.values(), word_list):
            for val, word2 in zip(result, word_list):
                self.dist_matrix[word1][word2] = val
        #self.hparas = tep
        self.display_dist_matrix()
        print("build tree ... ")

    def word_simus(self, word, word_list, fun, return_dict):
        ret = []
        for word2 in word_list:
           ret.append(self.cal_sim_test( self.ui_vector_dict[word],
                                         self.ui_vector_dict[word2], fun))
        return_dict[word] = ret[:]

    def build_simularity_tree(self, node_list):
        while node_list.__len__()>1:
            # find largest simularity pos
            wid1 = None
            wid2 = None
            sid1 = None
            sid2 = None
            max_sim = -999999999999
            for i1 in range(0,node_list.__len__()):
                #print( node_list[i1].word_list[0])
                word1 = node_list[i1].word_list[0]
                for i2 in range(i1 + 1,node_list.__len__()):
                    word2 = node_list[i2].word_list[0]

                    if i1 == i2:
                        continue

                    if word1 == word2:
                        print("error in build sim tree")
                        continue

                    if max_sim < self.dist_matrix[word1][word2]:
                        max_sim = self.dist_matrix[word1][word2]
                        wid1, wid2 = word1, word2
                        sid1, sid2 = i1, i2
                        #print(sid1, sid2)

            top_node = self.merge_by_sim(node_list[sid1],node_list[sid2])

            # update dist matrix
            if set(node_list[sid1].word_list) & set(node_list[sid2].word_list):
                print("merge error")

            new_dist = []
            for word1 in self.dist_matrix.keys():
                simus = [] #
                for word2 in (node_list[sid1].word_list + node_list[sid2].word_list):
                    simus.append(self.dist_matrix[word1][word2])

                if self.hparas.simu_metric == "min":
                    new_dist.append(min(simus))
                elif self.hparas.simu_metric == "max":
                    new_dist.append(max(simus))

            for word1 in (node_list[sid1].word_list + node_list[sid2].word_list):
                for idx, word2 in enumerate(self.dist_matrix):
                    self.dist_matrix[word1][word2] = new_dist[idx]
                    self.dist_matrix[word2][word1] = new_dist[idx]

            # update node list
            if sid1<sid2:
                node_list.pop(sid2)
                node_list.pop(sid1)
            elif sid1>sid2:
                node_list.pop(sid1)
                node_list.pop(sid2)
            else:
                raise RuntimeError('sid1 should not be equal to sid2')
            node_list.insert(0,top_node)
        self.root = node_list[0]   

    def build_huffman_tree(self, node_list):
        while node_list.__len__()>1:
            i1 = 0  # i1
            i2 = 1  # i2 
            if node_list[i2].freq < node_list[i1].freq :
                [i1,i2] = [i2,i1]
            for i in range(2,node_list.__len__()): # 
                if node_list[i].freq <node_list[i2].freq :
                    i2 = i
                    if node_list[i2].freq < node_list[i1].freq :
                        [i1,i2] = [i2,i1]
            top_node = self.merge_by_freq(node_list[i1],node_list[i2])
            if i1<i2:
                node_list.pop(i2)
                node_list.pop(i1)
            elif i1>i2:
                node_list.pop(i1)
                node_list.pop(i2)
            else:
                raise RuntimeError('i1 should not be equal to i2')
            node_list.insert(0,top_node)
        self.root = node_list[0]

    def generate_huffman_code(self, node, word_dict):
        # # use recursion in this edition
        # if node.left==None and node.right==None :
        #     word = node.value
        #     code = node.Huffman
        #     print(word,code)
        #     word_dict[word]['Huffman'] = code
        #     return -1
        #
        # code = node.Huffman
        # if code==None:
        #     code = ""
        # node.left.Huffman = code + "1"
        # node.right.Huffman = code + "0"
        # self.generate_huffman_code(node.left, word_dict)
        # self.generate_huffman_code(node.right, word_dict)

        # use stack but not recursion in this edition

        stack = [self.root]
        while (stack.__len__()>0):
            node = stack.pop()
            # go along left tree
            while node.left or node.right :
                path = node.path
                node.left.path = path + "1"
                node.right.path = path + "0"
                stack.append(node.right)
                node = node.left
            word = node.value
            path = node.path
            # print(word,'\t',code.__len__(),'\t',node.possibility)
            word_dict[word]['path'] = path

    def merge_by_freq(self,node1,node2):
        top_pos = node1.freq + node2.freq
        top_node = GeneralTreeNode(np.zeros([1,self.hparas.embedding_dim]), top_pos)
        #top_node = GeneralTreeNode(st.truncnorm.rvs(-1, 1, size=[1,vec_len])), top_pos)
        if node1.freq >= node2.freq :
            top_node.left = node1
            top_node.right = node2
        else:
            top_node.left = node2
            top_node.right = node1
        return top_node

    def merge_by_sim(self,node1,node2):
        top_pos = node1.freq + node2.freq
        top_node = GeneralTreeNode(np.zeros([1,self.hparas.embedding_dim]), top_pos, node1.word_list + node2.word_list)
        if node1.freq >= node2.freq :
            top_node.left = node1
            top_node.right = node2
        else:
            top_node.left = node2
            top_node.right = node1
        return top_node

    def test(self):
        print("hello")

    def display_dist_matrix(self):
        for w1 in self.dist_matrix:
            for w2 in self.dist_matrix[w1]:
                print(self.dist_matrix[w1][w2], end=' ')
            print("")
        print("")

    def cal_sim(self, v1, v2):
        if self.hparas.simu_func == "cos":
            return (np.inner(v1,v2)/(norm(v1)*norm(v2)))[0][0] if norm(v1)*norm(v2) > 0 else 0.0
        elif self.hparas.simu_func == "mht":
            return -1 * sum(abs(a-b) for a,b in zip(v1[0], v2[0]))
        elif self.hparas.simu_func == "euc":
            return -1 * sqrt(sum(pow(a-b, 2) for a,b in zip(v1[0], v2[0])))
        elif self.hparas.simu_func == "jaccard":
            q = r = s = 0
            for i in range(0, len(v1[0])):
                if v1[0][i] == 1 and v2[0][i] == 1:
                    q += 1
                elif v1[0][i] == 1:
                    r += 1
                elif v2[0][i] == 1:
                    s += 1
            return q/(q + r + s)


    def cal_sim_test(self, v1, v2, fun):
        if fun == "cos":
            return (np.inner(v1,v2)/(norm(v1)*norm(v2)))[0][0] if norm(v1)*norm(v2) > 0 else 0.0
        elif fun == "mht":
            return -1 * sum(abs(a-b) for a,b in zip(v1[0], v2[0]))
        elif fun == "euc":
            return -1 * sqrt(sum(pow(a-b, 2) for a,b in zip(v1[0], v2[0])))
        elif fun == "jaccard":
            q = r = s = 0
            for i in range(0, len(v1[0])):
                if v1[0][i] == 1 and v2[0][i] == 1:
                    q += 1
                elif v1[0][i] == 1:
                    r += 1
                elif v2[0][i] == 1:
                    s += 1
            return q/(q + r + s)
