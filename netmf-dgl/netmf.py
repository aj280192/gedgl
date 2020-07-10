import dgl
import time
import argparse
import logging
import numpy as np
import scipy.sparse as sp

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
        

class NetMF():
    
    def __init__(self, dimension, window_size, rank, negative, is_large=False):
        
        self.dimension = dimension
        self.window_size = window_size
        self.rank = rank
        self.negative = negative
        self.is_large = is_large
        
    def train(self, G):
        A = G.adjacency_matrix_scipy()
        if not self.is_large:
            print("Running NetMF for a small window size...")
            deepwalk_matrix = self._compute_deepwalk_matrix(
                A, window=self.window_size, b=self.negative
            )

        else:
            print("Running NetMF for a large window size...")
            vol = float(A.sum())
            evals, D_rt_invU = self._approximate_normalized_laplacian(
                A, rank=self.rank, which="LA"
            )
            deepwalk_matrix = self._approximate_deepwalk_matrix(
                evals, D_rt_invU, window=self.window_size, vol=vol, b=self.negative
            )
        # factorize deepwalk matrix with SVD
        u, s, _ = sp.linalg.svds(deepwalk_matrix, self.dimension)
        self.embeddings = sp.diags(np.sqrt(s)).dot(u.T).T
        return self.embeddings
    
    def _compute_deepwalk_matrix(self, A, window, b):
        # directly compute deepwalk matrix
        n = A.shape[0]
        vol = float(A.sum())
        L, d_rt = sp.csgraph.laplacian(A, normed=True, return_diag=True)
        # X = D^{-1/2} A D^{-1/2}
        X = sp.identity(n) - L
        S = np.zeros_like(X)
        X_power = sp.identity(n)
        for i in range(window):
            print("Compute matrix %d-th power" % (i + 1))
            X_power = X_power.dot(X)
            S += X_power
        S *= vol / window / b
        D_rt_inv = sp.diags(d_rt ** -1)
        M = D_rt_inv.dot(D_rt_inv.dot(S).T).todense()
        M[M <= 1] = 1
        Y = np.log(M)
        return sp.csr_matrix(Y)
    
    def _approximate_normalized_laplacian(self, A, rank, which="LA"):
        # perform eigen-decomposition of D^{-1/2} A D^{-1/2} and keep top rank eigenpairs
        n = A.shape[0]
        L, d_rt = sp.csgraph.laplacian(A, normed=True, return_diag=True)
        # X = D^{-1/2} W D^{-1/2}
        X = sp.identity(n) - L
        print("Eigen decomposition...")
        evals, evecs = sp.linalg.eigsh(X, rank, which=which)
        print(
            "Maximum eigenvalue %f, minimum eigenvalue %f" % (np.max(evals), np.min(evals))
        )
        print("Computing D^{-1/2}U..")
        D_rt_inv = sp.diags(d_rt ** -1)
        D_rt_invU = D_rt_inv.dot(evecs)
        return evals, D_rt_invU
    
    def _deepwalk_filter(self, evals, window):
        for i in range(len(evals)):
            x = evals[i]
            evals[i] = 1.0 if x >= 1 else x * (1 - x ** window) / (1 - x) / window
        evals = np.maximum(evals, 0)
        print(
            "After filtering, max eigenvalue=%f, min eigenvalue=%f" %
            (np.max(evals),
            np.min(evals))
        )
        return evals

    def _approximate_deepwalk_matrix(self, evals, D_rt_invU, window, vol, b):
        # approximate deepwalk matrix
        evals = self._deepwalk_filter(evals, window=window)
        X = sp.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
        M = X.dot(X.T) * vol / b
        M[M <= 1] = 1
        Y = np.log(M)
        print("Computed DeepWalk matrix with %d non-zero elements" % np.count_nonzero(Y))
        return sp.csr_matrix(Y)
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
            help="input file path")
    parser.add_argument("--output", type=str, required=True,
            help="embedding output file path")
    parser.add_argument("--rank", default=256, type=int,
            help="#eigenpairs used to approximate normalized graph laplacian.")
    parser.add_argument("--dim", default=128, type=int,
            help="dimension of embedding")
    parser.add_argument("--window", default=10,
            type=int, help="context window size")
    parser.add_argument("--negative", default=1.0, type=float,
            help="negative sampling")
    parser.add_argument('--islarge', type=bool, default=True,
            help="using netmf for large window size")


    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(message)s') # include timestamp
    
    print('Make sure embedding dimension is between 1 to |V| (number of nodes of the graph)')
    
    net, node2id, id2node, netsm = ReadTxtNet(args.input)
    G = net2graph(netsm)
    
    start_time = time.time()
    model = NetMF(args.dim,args.window,args.rank,args.negative,args.islarge)
    embedding = model.train(G)
    print("Total used time: %.2f" % (time.time() - start_time))
    
    save_embedding(args.output,embedding,id2node)
    
