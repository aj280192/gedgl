import torch
import torch.multiprocessing as mp

import numpy as np
import scipy.sparse as sp
import random
import time
import dgl

import torch.nn as nn
from torch.nn import init
from math import log

import argparse

def ReadTxtNet(file_path="", undirected=True):
    """ Read the txt network file """ 

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

def node_edge_list(G):
    """ Function returns node and edge list of graph """
    nodes = list(map(int,G.nodes()))
    edges = {}
    for node in nodes:
        edges[node] = list(map(int,G.successors(node)))
    return nodes, edges


class Verse(nn.Module):
    """ Negative sampling based skip-gram """
    def __init__(self, 
        nodes,
        edges,
        emb_dimension,
        epochs,
        negative,
        lr,
        alpha,
        worker
        ):
        """ initialize embedding on CPU 
        Paremeters
        ----------
        nodes list : list of nodes in graph
        edges list : list of edges in graph
        num_nodes int : number of nodes in graph
        emb_dimension int : embedding dimension
        epoch int : number of iteration to perform training
        negative int : number of negative samples per positive sample
        lr float : initial learning rate
        alpha float : restart factor
        worker : number of processes for parallel processing
        """
        super(Verse, self).__init__()
        self.nodes = nodes
        self.edges = edges
        self.num_nodes = len(nodes)
        self.emb_dimension = emb_dimension
        self.epochs = epochs 
        self.negative = negative
        self.lr = lr
        self.alpha = alpha
        self.worker = worker
        self.nce_bias = log(self.num_nodes)
        self.nce_neg_bias = log(self.num_nodes/negative)

        # content embedding
        self.W = nn.Embedding(
            self.num_nodes, self.emb_dimension, sparse=True)
        
        # initialze embedding
        initrange = 0.5
        init.uniform_(self.W.weight.data, -initrange, initrange)


        # lookup_table is used for fast sigmoid computing
        self.lookup_table = torch.sigmoid(torch.arange(-6.01, 6.01, 0.01))
        self.lookup_table[0] = 0.
        self.lookup_table[-1] = 1.

    def fast_sigmoid(self, score):
        """ do fast sigmoid by looking up in a pre-defined table """
        idx = torch.floor((score + 6.01) / 0.01).long()
        return self.lookup_table[idx]

    def train(self):
        """ fast learning with auto grad off
        """
        for i in range(int(self.epochs/self.worker)):
            with torch.no_grad():
                lr = self.lr
                nce_bias = self.nce_bias
                nce_neg_bias = self.nce_neg_bias
                
                # genarating positive and negative indexes
                idx_pos_u, idx_pos_v, idx_neg_u, idx_neg_v = self.sample_index() # generating the samples for current epoch
                

                # positive 
                emb_pos_u = self.W(idx_pos_u).to('cpu')
                emb_pos_v = self.W(idx_pos_v).to('cpu')


                pos_score = torch.sum(torch.mul(emb_pos_u,emb_pos_v), dim=1) - nce_bias
                pos_score = torch.clamp(pos_score, max = 6, min = -6)
                score = ((1 - self.fast_sigmoid(pos_score)) * lr).unsqueeze(1)
                grad_u_pos = score * emb_pos_v 
                grad_v_pos = score * emb_pos_u

                self.W.weight.data.index_add_(0,idx_pos_u,grad_u_pos)
                self.W.weight.data.index_add_(0,idx_pos_v,grad_v_pos)


                # negatives
                emb_neg_u = self.W(idx_neg_u).to('cpu')
                emb_neg_v = self.W(idx_neg_v).to('cpu')


                neg_score = torch.sum(torch.mul(emb_neg_u,emb_neg_v), dim=1) - nce_neg_bias
                neg_score = torch.clamp(neg_score, max = 6, min = -6)
                score = (0 - self.fast_sigmoid(neg_score) * lr).unsqueeze(1)
                grad_u_neg = score * emb_neg_v 
                grad_v_neg = score * emb_neg_u

                self.W.weight.data.index_add_(0,idx_neg_u,grad_u_neg)
                self.W.weight.data.index_add_(0,idx_neg_v,grad_v_neg)

        return

    def save_embedding_txt(self, id2node, file_name):
        """ Write embedding to local file. For future use.
        Parameter
        ---------
        dataset DeepwalkDataset : the dataset
        file_name str : the file name
        """
        embedding = self.W.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (self.num_nodes, self.emb_dimension))
            for wid in range(self.num_nodes):
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (str(id2node[wid]), e))
    
    def sample_neighbor(self,node):
        dval = random.uniform(0, 1)
        if dval < self.alpha:
            v = random.choice(self.edges[node])
        else:
            v = node
        return v

    def sample_index(self):
        num_nodes = self.num_nodes
        negatives = self.negative
        index_pos_u, index_pos_v, index_neg_u, index_neg_v = [], [], [], []

        for i in range(num_nodes): 
            
            u = random.randint(0,num_nodes-1)   # sampled node
            
            index_pos_u.append(u) # rsample u
            index_pos_v.append(self.sample_neighbor(u)) # positive sample v
            
            index_neg_u.extend([u] * negatives) # sample u added for negatives
            for j in range(negatives):
                index_neg_v.append(random.randint(0,num_nodes-1)) # negative sample v_


        index_pos_u = torch.LongTensor(index_pos_u)
        index_pos_v = torch.LongTensor(index_pos_v)

        index_neg_u = torch.LongTensor(index_neg_u)
        index_neg_v = torch.LongTensor(index_neg_v)    

        return index_pos_u,index_pos_v,index_neg_u,index_neg_v
    
    def train_mp(self):
        ps = []
        for i in range(self.worker):
            p = mp.Process(target=self.train())
            ps.append(p)
            p.start()
    
        for p in ps:
            p.join()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verse")
    parser.add_argument('--net_file', type=str, 
            help="path of the txt network file, builtin dataset include youtube-net and blog-net") 
    parser.add_argument('--output_file', type=str, default="emb.npy",
            help='path of the output npy embedding file')
    parser.add_argument('--dim', default=128, type=int, 
            help="embedding dimensions")
    parser.add_argument('--negative', default=5, type=int, 
            help="negative samples for each positve node pair")
    parser.add_argument('--epochs', default=1, type=int, 
            help="epochs")
    parser.add_argument('--lr', default=0.0025, type=float, 
            help="learning rate")
    parser.add_argument('--alpha', type=float, default=0.85,  
            help='alpha-value must be float less than 1')
    parser.add_argument('--worker', type=int, default=1,  
            help='number of workers for parallel processing')
    args = parser.parse_args()

    net,node2id,id2node,net_sm = ReadTxtNet(args.net_file)
    G = net2graph(net_sm)
    nodes,edges = node_edge_list(G)
    
    
    model = Verse(nodes,edges,args.dim,args.epochs,args.negative,args.lr,args.alpha,args.worker)
    if args.worker > 1:
       model.share_memory()
       model.train_mp()
    else:
       model.train()
    
    start_time = time.time()
    model.train_mp()
    print("Total used time for training: %.2f" % (time.time() - start_time))
    
    model.save_embedding_txt(id2node,args.output_file)