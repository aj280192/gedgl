import numpy as np
import scipy.sparse as sp
import pickle
import torch
from torch.utils.data import DataLoader
import random
import time
import dgl
from utils import shuffle_walks
np.random.seed(3141592653)

def ReadTxtNet(file_path="", undirected=True):
    """ Read the txt network file. 
    Notations: The network is unweighted.
    Parameters
    ----------
    file_path str : path of network file
    undirected bool : whether the edges are undirected
    Return
    ------
    net dict : a dict recording the connections in the graph
    node2id dict : a dict mapping the nodes to their embedding indices 
    id2node dict : a dict mapping nodes embedding indices to the nodes
    """
    node2id = {}
    id2node = {}
    cid = 0

    src = []
    dst = []
    net = {}
    with open(file_path, "r") as f:
        for line in f.readlines():
            n1, n2 = list(map(int, line.strip().split(" ")[:2]))
            if n1 not in node2id:
                node2id[n1] = cid
                id2node[cid] = n1
                cid += 1
            if n2 not in node2id:
                node2id[n2] = cid
                id2node[cid] = n2
                cid += 1

            n1 = node2id[n1]
            n2 = node2id[n2]
            if n1 not in net:
                net[n1] = {n2: 1}
                src.append(n1)
                dst.append(n2)
            elif n2 not in net[n1]:
                net[n1][n2] = 1
                src.append(n1)
                dst.append(n2)
            
            if undirected:
                if n2 not in net:
                    net[n2] = {n1: 1}
                    src.append(n2)
                    dst.append(n1)
                elif n1 not in net[n2]:
                    net[n2][n1] = 1
                    src.append(n2)
                    dst.append(n1)

    print("node num: %d" % len(net))
    print("edge num: %d" % len(src))
    assert max(net.keys()) == len(net) - 1, "error reading net, quit"

    sm = sp.coo_matrix(
        (np.ones(len(src)), (src, dst)),
        dtype=np.float32)

    return net, node2id, id2node, sm

def net2graph(net_sm):
    """ Transform the network to DGL graph
    Return 
    ------
    G DGLGraph : graph by DGL
    """
    start = time.time()
    G = dgl.DGLGraph(net_sm)
    end = time.time()
    t = end - start
    print("Building DGLGraph in %.2fs" % t)
    return G

class Node2vecDataset:
    def __init__(self, 
            net_file,
            map_file,
            p,
            q,
            walk_length=80,
            window_size=5,
            num_walks=10,
            batch_size=32,
            negative=5,
            num_procs=4,
            fast_neg=True,
            ):
        """ This class has the following functions:
        1. Transform the txt network file into DGL graph;
        2. Generate random walk sequences for the trainer;
        3. Provide the negative table if the user hopes to sample negative
        nodes according to nodes' degrees;
        Parameter
        ---------
        net_file str : path of the txt network file
        walk_length int : number of nodes in a sequence
        window_size int : context window size
        num_walks int : number of walks for each node
        batch_size int : number of node sequences in each batch
        negative int : negative samples for each positve node pair
        fast_neg bool : whether do negative sampling inside a batch
        """
        self.walk_length = walk_length
        self.window_size = window_size
        self.num_walks = num_walks
        self.batch_size = batch_size
        self.negative = negative
        self.num_procs = num_procs
        self.fast_neg = fast_neg
        self.net, self.node2id, self.id2node, self.sm = ReadTxtNet(net_file)
        self.save_mapping(map_file)
        self.G = net2graph(self.sm)
        self.p = p
        self.q = q
        
        # calculating the transition probability
        start = time.time()
        self.preprocess_transition_probs()
        print("transition probability calculated in %.2fs" % (time.time() - start))

        # random walk seeds
        start = time.time()
        seeds = torch.cat([torch.LongTensor(self.G.nodes())] * num_walks)
        self.seeds = torch.split(shuffle_walks(seeds), int(np.ceil(len(self.net) * self.num_walks / self.num_procs)), 0)
        end = time.time()
        t = end - start
        print("%d seeds in %.2fs" % (len(seeds), t))

        # negative table for true negative sampling
        if not fast_neg:
            node_degree = np.array(list(map(lambda x: len(self.net[x]), self.net.keys())))
            node_degree = np.power(node_degree, 0.75)
            node_degree /= np.sum(node_degree)
            node_degree = np.array(node_degree * 1e8, dtype=np.int)
            self.neg_table = []
            for idx, node in enumerate(self.net.keys()):
                self.neg_table += [node] * node_degree[idx]
            self.neg_table_size = len(self.neg_table)
            self.neg_table = np.array(self.neg_table, dtype=np.long)
            del node_degree
        
    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge with uniform weights.
        '''
        G = self.G
        p = self.p
        q = self.q
        unnormalized_probs = []
        
        for dst_nbr in sorted(G.successors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(1/p)
            elif G.has_edges_between(dst_nbr, src):
                unnormalized_probs.append(1)
            else:
                unnormalized_probs.append(1/q)
            
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            
        return self.alias_setup(normalized_probs)
    
    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks with uniform weight for edges.
        '''
        
        G = self.G
        
        alias_nodes = {}
        
        for n in G.nodes():
            unnormalized_probs_dgl = [1 for _ in G.successors(n)]
            norm_const_dgl = sum(unnormalized_probs_dgl)
            normalized_probs_dgl =  [float(u_prob)/norm_const_dgl for u_prob in unnormalized_probs_dgl]
            alias_nodes[int(n)] = self.alias_setup(normalized_probs_dgl)
        
        alias_edges = {}
        
        for src,dst in zip(G.edges()[0],G.edges()[1]):
            alias_edges[(int(src),int(dst))]= self.get_alias_edge(src,dst)
            
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        
        return


    def alias_setup(self,probs):
        '''
        Compute utility lists for non-uniform sampling from discrete distributions.
        Refer to https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
        for details
        '''
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)
        smaller = []
        larger = []
        
        for kk, prob in enumerate(probs):
            q[kk] = K*prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
        
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            
            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        
        return J, q

    def create_sampler(self, gpu_id):
        """
        Using alias random sampling to sample the nodes and generate second order travesal of the graph.
        """
        return DeepwalkSampler(self.G, self.seeds[gpu_id], self.walk_length,self.alias_nodes,self.alias_edges,self.p,self.q)

    def save_mapping(self, map_file):
        with open(map_file, "wb") as f:
            pickle.dump(self.node2id, f)

class DeepwalkSampler(object):
    def __init__(self, G, seeds, walk_length,alias_nodes,alias_edges,p,q):
        self.G = G
        self.seeds = seeds
        self.walk_length = walk_length
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        self.p = p
        self.q = q
    
    def sample(self, seeds):
        '''
        Repeatedly simulate random walks from each node.
        '''
        walks = []
        
        for node in seeds:
            node_walks = []
            node_walks.append(self.node2vec_walk(int(node)))
            walks.append(node_walks)
        
        return torch.tensor(walks)
    
    def node2vec_walk(self,start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        G = self.G
        
        walk_length = self.walk_length
        
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted([int(x) for x in G.successors(cur)])
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[self.alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    nextnode = cur_nbrs[self.alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(nextnode)
            else:
                break
        
        return walk
    
    def alias_draw(self,J, q):
        '''
        Draw sample from a non-uniform discrete distribution using alias sampling.
        '''
        K = len(J)
        ran = np.random.rand()
        
        kk = int(np.floor(ran*K))
        
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]

