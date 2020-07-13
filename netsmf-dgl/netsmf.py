import dgl
import argparse
import logging
import numpy as np
import scipy.sparse as sp
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd
from multiprocessing import Pool
import time


def ReadTxtNet(file_path="", undirected=True):
    """ Read the txt network file."""

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

def save_embedding(filepath,embedding,id2node):
    num_nodes, dimension = embedding.shape
    with open(filepath,'w') as f:
        f.write('%d %d\n' % (num_nodes, dimension))
        for wid in range(num_nodes):
            e = ' '.join(map(lambda x: str(x), embedding[wid]))
            f.write('%s %s\n' % (str(id2node[wid]), e))


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
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
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)
    
    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

class NetSMF():
    def __init__(self, dimension, window_size, negative, num_round, worker):
        self.dimension = dimension
        self.window_size = window_size
        self.negative = negative
        self.worker = worker
        self.num_round = num_round

    def train(self, G):
        self.G = G
        self.is_directed = True
        self.num_node = self.G.number_of_nodes()
        self.num_edge = self.G.number_of_edges()
        self.edges = [[int(src), int(dst)] for src,dst in zip(self.G.edges()[0], self.G.edges()[1])]


        self.num_neigh = np.asarray([len(list(self.G.successors(i))) for i in range(self.num_node)])
        
        self.neighbors =  [[int(v) for v in self.G.successors(i)] for i in range(self.num_node)]
        
        s = time.time()
        self.alias_nodes = {}
        self.node_weight = {}
        
        for i in range(self.num_node):
            
            
            unnormalized_probs = [1 for _ in self.G.successors(i)] # defaulting the weight for graph to 1
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            self.alias_nodes[i] = alias_setup(normalized_probs)
            self.node_weight[i] = dict(zip([int(nbr) for nbr in self.G.successors(i)],unnormalized_probs,))

        t = time.time()
        print("alias_nodes", t - s)

        # run netsmf algorithm with multiprocessing and apply randomized svd
        print(
            "number of sample edges ", self.num_round * self.num_edge * self.window_size
        )
        print("random walk start...")
        t0 = time.time()
        results = []
        pool = Pool(processes=self.worker)
        for i in range(self.worker):
            results.append(pool.apply_async(func=self._random_walk_matrix, args=(i,)))
        pool.close()
        pool.join()
        print("random walk time", time.time() - t0)

        matrix = sp.lil_matrix((self.num_node, self.num_node))
        
        t1 = time.time()
        for res in results:
            matrix += res.get()
        print("number of nzz", matrix.nnz)
        
        t2 = time.time()
        print("construct random walk matrix time", time.time() - t1)
        
        del results
        
        A = self.G.adjacency_matrix_scipy()
        degree = sp.diags(np.array(A.sum(axis=0))[0], format="csr")
        degree_inv = degree.power(-1)


        L = sp.csgraph.laplacian(matrix, normed=False, return_diag=False)
        M = degree_inv.dot(degree - L).dot(degree_inv)
        M = M * A.sum() / self.negative
        M.data[M.data <= 1] = 1
        M.data = np.log(M.data)
        print("construct matrix sparsifier time", time.time() - t2)

        embedding = self._get_embedding_rand(M)
        return embedding

    def _get_embedding_rand(self, matrix):
        # Sparse randomized tSVD for fast embedding
        t1 = time.time()
        l = matrix.shape[0]
        smat = sp.csc_matrix(matrix)
        print("svd sparse", smat.data.shape[0] * 1.0 / l ** 2)
        U, Sigma, VT = randomized_svd(
            smat, n_components=self.dimension, n_iter=5, random_state=None
        )
        U = U * np.sqrt(Sigma)
        U = preprocessing.normalize(U, "l2")
        print("sparsesvd time", time.time() - t1)
        return U

    def _path_sampling(self, u, v, r):
        # sample a r-length path from edge(u, v) and return path end node
        k = np.random.randint(r) + 1
        zp, rand_u, rand_v = 1e-20, k - 1, r - k
        for i in range(rand_u):
            new_u = self.neighbors[u][
                alias_draw(self.alias_nodes[u][0], self.alias_nodes[u][1])
            ]
            zp += 2.0 / self.node_weight[u][new_u]
            u = new_u
        for j in range(rand_v):
            new_v = self.neighbors[v][
                alias_draw(self.alias_nodes[v][0], self.alias_nodes[v][1])
            ]
            zp += 2.0 / self.node_weight[v][new_v]
            v = new_v
        return u, v, zp

    def _random_walk_matrix(self, pid):
        # construct matrix based on random walk
        np.random.seed(pid)
        matrix = sp.lil_matrix((self.num_node, self.num_node))
        #t0 = time.time()
        for round in range(int(self.num_round / self.worker)):
            #if round % 10 == 0 and pid == 0:
            #    print(
            #        "round %d / %d, time: %lf"
            #        % (round * self.worker, self.num_round, time.time() - t0)
            #    )
            for i in range(self.num_edge):
                u, v = self.edges[i]
                #if not self.is_directed and np.random.rand() > 0.5:
                #   v, u = self.edges[i]
                for r in range(1, self.window_size + 1):
                    u_, v_, zp = self._path_sampling(u, v, r)
                    matrix[u_, v_] += 2 * r / self.window_size / self.num_round / zp
        return matrix


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
            help="input file path")
    parser.add_argument("--output", type=str, required=True,
            help="embedding output file path")
    parser.add_argument("--dim", default=128, type=int,
            help="dimension of embedding")
    parser.add_argument("--window", default=10,
            type=int, help="Window size of approximate matrix. Default is 10.")
    parser.add_argument("--negative", default=1.0, type=float,
            help="negative sampling")
    parser.add_argument('--num-round', type=int, default=100,
            help="Number of round in NetSMF. Default is 100.")
    parser.add_argument("--worker", default=10, type=int,
            help="number of workers for multiprocessing")


    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(message)s') # include timestamp
    
    print('Make sure embedding dimension is between 1 to |V| (number of nodes of the graph)')
    
    net, node2id, id2node, netsm = ReadTxtNet(args.input)
    G = net2graph(netsm)
    
    start_time = time.time()
    model = NetSMF(args.dim,args.window,args.negative,args.num_round,args.worker)
    embedding = model.train(G)
    print("Total used time: %.2f" % (time.time() - start_time))
    
    save_embedding(args.output,embedding,id2node)
    