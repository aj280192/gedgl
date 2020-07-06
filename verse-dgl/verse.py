import torch
from functools import wraps
from _thread import start_new_thread
import torch.multiprocessing as mp

import os
import numpy as np
import scipy.sparse as sp
from dgl.data.utils import download, _get_dgl_url, get_download_dir, extract_archive
import random
import time
import dgl

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from math import log

import argparse
import dgl
import torch.multiprocessing as mp
import time


def thread_wrapped_func(func):
    """Wrapped func for torch.multiprocessing.Process.
    With this wrapper we can use OMP threads in subprocesses
    otherwise, OMP_NUM_THREADS=1 is mandatory.
    How to use:
    @thread_wrapped_func
    def func_to_wrap(args ...):
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = mp.Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function

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
    if file_path == 'youtube' or file_path == 'blog':
        name = file_path
        dir = get_download_dir()
        zip_file_path='{}/{}.zip'.format(dir, name)
        download(_get_dgl_url(os.path.join('dataset/DeepWalk/', '{}.zip'.format(file_path))), path=zip_file_path)
        extract_archive(zip_file_path,
                        '{}/{}'.format(dir, name))
        file_path = "{}/{}/{}-net.txt".format(dir, name, name)

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
        only_gpu,
        only_cpu,
        mix
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
        self.nce_bias = log(self.num_nodes)
        self.nce_neg_bias = log(self.num_nodes/negative)
        self.train_count = self.num_nodes * epochs
        self.only_cpu = only_cpu
        self.only_gpu = only_gpu
        self.mixed_train = mix
        
        print("train count : {}".format(self.train_count))
        
        # initialize the device as cpu
        self.device = torch.device("cpu")

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

    def share_memory(self):
        """ share the parameters across subprocesses """
        self.W.weight.share_memory_()

    def set_device(self, gpu_id):
        """ set gpu device """
        self.device = torch.device("cuda:%d" % gpu_id)
        print("The device is", self.device)
        self.lookup_table = self.lookup_table.to(self.device)


    def all_to_device(self, gpu_id):
        """ move all of the parameters to a single GPU """
        self.device = torch.device("cuda:%d" % gpu_id)
        self.set_device(gpu_id)
        self.W = self.W.cuda(gpu_id)

    def fast_sigmoid(self, score):
        """ do fast sigmoid by looking up in a pre-defined table """
        idx = torch.floor((score + 6.01) / 0.01).long()
        return self.lookup_table[idx]

    def fast_learn(self):
        """ fast learning with auto grad off
        """
        print("inside fast learn")
        lr = self.lr
        nce_bias = self.nce_bias
        nce_neg_bias = self.nce_neg_bias
        negative = self.negative
        count = 0
        train_count = self.train_count
        while (True):
            if (count > train_count):
                break
            u_node = random.choice(self.nodes)
            v_node = self.sample_neighbor(u_node)
            self.update(u_node,v_node, 1, nce_bias, lr)
            for i in range(negative):
                v_neg_node = random.choice(self.nodes)
                self.update(u_node,v_neg_node, 0, nce_neg_bias, lr)
            count += 1
        return
    
    def update(self, u, v, label, bias, lr):
        score = -bias
        W_u = self.W.weight[u]
        W_v = self.W.weight[v]
        score += torch.mul(W_u,W_v)
        score = torch.clamp(score, max=6, min=-6)
        score = (label - self.fast_sigmoid(score)) * lr
        self.W.weight[u] += W_v * score
        self.W.weight[v] += W_u * score
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
    
class Trainer:
    def __init__(self, args):
        """ Initializing the trainer with the input arguments """
        self.args = args
        self.net, self.node2id, self.id2node, self.sm = ReadTxtNet(args.net_file)
        self.G = net2graph(self.sm)
        self.nodes, self.edges = self.node_edge_list()
        self.emb_model = None

    def init_device_emb(self):
        """ set the device before training 
        will be called once in fast_train_mp / fast_train
        """
        choices = sum([self.args.only_gpu, self.args.only_cpu, self.args.mix])
        assert choices == 1, "Must choose only *one* training mode in [only_cpu, only_gpu, mix]"
        
        # initializing embedding on CPU
        self.emb_model = Verse(
            nodes=self.nodes, 
            edges=self.edges,
            emb_dimension=self.args.dim,
            epochs = self.args.epochs,
            negative= self.args.negative,
            lr = self.args.lr,
            alpha = self.args.alpha,
            only_cpu=self.args.only_cpu,
            only_gpu=self.args.only_gpu,
            mix=self.args.mix
            )
        
        torch.set_num_threads(self.args.num_threads)
        if self.args.only_gpu:
            print("Run in 1 GPU")
            assert self.args.gpus[0] >= 0
            self.emb_model.all_to_device(self.args.gpus[0])
        elif self.args.mix:
            print("Mix CPU with %d GPU" % len(self.args.gpus))
            if len(self.args.gpus) == 1:
                assert self.args.gpus[0] >= 0, 'mix CPU with GPU should have avaliable GPU'
                self.emb_model.set_device(self.args.gpus[0])
        else:
            print("Run in CPU process")
            self.args.gpus = [torch.device('cpu')]


    def train(self):
        """ train the embedding """
        if len(self.args.gpus) > 1:
            self.fast_train_mp()
        else:
            self.fast_train()

    def fast_train_mp(self):
        """ multi-cpu-core or mix cpu & multi-gpu """
        self.init_device_emb()
        self.emb_model.share_memory()

        start_all = time.time()
        ps = []

        for i in range(len(self.args.gpus)):
            p = mp.Process(target=self.fast_train_sp, args=(self.args.gpus[i],))
            ps.append(p)
            p.start()

        for p in ps:
            p.join()
        
        print("Used time: %.2fs" % (time.time()-start_all))
        
        self.emb_model.save_embedding_txt(self.id2node, self.args.output_emb_file)

    @thread_wrapped_func
    def fast_train_sp(self, gpu_id):
        """ a subprocess for fast_train_mp """
        if self.args.mix:
            self.emb_model.set_device(gpu_id)
        torch.set_num_threads(self.args.num_threads)
        
        start = time.time()
        with torch.no_grad():
            self.emb_model.fast_learn()
            

    def fast_train(self):
        """ fast train with dataloader """
        self.init_device_emb()
        print("inside train!!")

        start_all = time.time()
        with torch.no_grad():
            print("calling train method of modal!!")
            self.emb_model.fast_learn()
            
        print("Training used time: %.2fs" % (time.time()-start_all))
        
        self.emb_model.save_embedding_txt(self.id2node, self.args.output_file)
    
    def node_edge_list(self):
        G = self.G
        nodes = list(G.nodes())
        edges = {}
        for node in nodes:
            edges[node] = G.successors(node)
        del G
        return nodes, edges

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
    parser.add_argument('--lr', default=0.2, type=float, 
            help="learning rate")
    parser.add_argument('--mix', default=False, action="store_true", 
            help="mixed training with CPU and GPU")
    parser.add_argument('--only_cpu', default=False, action="store_true", 
            help="training with CPU")
    parser.add_argument('--only_gpu', default=False, action="store_true", 
            help="training with GPU")
    parser.add_argument('--num_threads', default=2, type=int, 
            help="number of threads used for each CPU-core/GPU")
    parser.add_argument('--gpus', type=int, default=[-1], nargs='+', 
            help='a list of active gpu ids, e.g. 0')
    parser.add_argument('--alpha', type=float, default=0.85,  
            help='alpha-value must be float less than 1')
    args = parser.parse_args()

    start_time = time.time()
    trainer = Trainer(args)
    trainer.train()
    print("Total used time: %.2f" % (time.time() - start_time))