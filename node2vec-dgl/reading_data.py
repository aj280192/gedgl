import numpy as np
import dgl
import torch
import random
import time
np.random.seed(31415926)

def ReadTxtNet(file_path=""):
    node2id = {}
    id2node = {}
    cid = 0

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
            try:
                net[n1][n2] = 1
            except:
                net[n1] = {n2: 1}
            try:
                net[n2][n1] = 1
            except:
                net[n2] = {n1: 1}

    print("node num: %d" % len(net))
    print("edge num: %d" % (sum(list(map(lambda i: len(net[i]), net.keys())))/2))
    if max(net.keys()) != len(net) - 1:
        print("error reading net, quit")
        exit(1)
    return net, node2id, id2node

def net2graph(net):
    G = dgl.DGLGraph()
    G.add_nodes(len(net))
    for i in net:
        G.add_edges(i, list(net[i].keys()))
    return G

class Node2vecDataset:
    def __init__(self, args):
        self.walk_length = args.walk_length
        self.window_size = args.window_size
        self.num_walks = args.num_walks
        self.batch_size = args.batch_size
        self.negative = args.negative
        self.net, self.node2id, self.id2node = ReadTxtNet(args.net_file)
        self.G = net2graph(self.net)
        self.walks = []
        self.p = args.p
        self.q = args.q
        self.rand = random.Random()
        
        # setting the transition probabilities
        self.preprocess_transition_probs()

        # random walk using alias sampling method!!
        start = time.time()
        walks = torch.tensor(self.simulate_walks())
        self.walks = list(walks.view(-1, self.walk_length))
        end = time.time()
        t = end - start
        print("%d walks in %.2fs" % (len(self.walks), t))

        # negative table for true negative sampling
        if not args.fast_neg:
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
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break
        
        return walk
    
    def simulate_walks(self):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        num_walks = self.num_walks
        
        walks = []
        nodes = [int(x) for x in G.nodes()]
        
        for node in nodes:
            node_walks = []
            for walk_iter in range(num_walks):
                node_walks.append(self.node2vec_walk(node))
            walks.append(node_walks)
        
        return walks
    
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
        
        return alias_setup(normalized_probs)
    
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
            alias_nodes[int(n)] = alias_setup(normalized_probs_dgl)
        
        alias_edges = {}
        
        for src,dst in zip(G.edges()[0],G.edges()[1]):
            alias_edges[(int(src),int(dst))]= self.get_alias_edge(src,dst)
            
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        
        return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
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

def alias_draw(J, q):
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