import numpy as np
import random
import argparse
import logging
from sklearn import linear_model
from sklearn import metrics

def data_preprocessing(emb_file,edge_file,neg_file):
    data = {}
    with open(emb_file) as e:
        for line in e.readlines()[1:]: #skipping the first line of embedding file
            values = line.strip().split(' ')
            data[int(values[0])] = list(map(float,values[1:]))
            
    print("embedding value loaded!!")
    
    true_edges = []
    with open(edge_file) as te:
        for line in te.readlines():
            src, dst = line.strip().split(' ')
            true_edges.append((int(src),int(dst)))
    
    print("True edge list created!!")
    
    neg_edges = []
    with open(neg_file) as ne:
        for line in ne.readlines():
            node, label = line.strip().split(' ')
            neg_edges.append((int(src),int(dst)))
    
    return data, true_edges, neg_edges

def random_split(true_edges,neg_edges,split):
    
    true_len = len(true_edges)
    true_split = int(true_len * split)
    neg_len = len(neg_edges)
    neg_split = int(neg_len * split)
    
    random.shuffle(true_edges)
    random.shuffle(neg_edges)
    
    true_test  = true_edges[:true_split]
    true_train = true_edges[true_split:]
    
    neg_test  = neg_edges[:neg_split]
    neg_train = neg_edges[neg_split:]
    
    return true_train,true_test,neg_train,neg_test


def model_data(true_edge,neg_edge,embedding):
    
    X = []
    y = []
    for n1, n2 in true_edge:
        node1 = np.array(embedding[n1])
        node2 = np.array(embedding[n2])
        edge_data = np.multiply(node1,node2)
        X.append(edge_data)
        y.append(1)
    
    for n1, n2 in neg_edge:
        node1 = np.array(embedding[n1])
        node2 = np.array(embedding[n2])
        edge_data = np.multiply(node1,node2)
        X.append(edge_data)
        y.append(0)
    
    X = np.array(X)
    y = np.array(y)
    
    return X,y

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", type=str, required=True,
            help="embedding file path")
    parser.add_argument("--edge", type=str, required=True,
            help="edge list file path")
    parser.add_argument("--neg", type=str, required=True,
            help="negative edge list")
    parser.add_argument("--split", type=float, default=0.4,
            help="train & test split")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(message)s')
    
    embedding, pos_edges, neg_edges = data_preprocessing(args.emb,args.edge, args.neg)
    
    true_train, true_test, neg_train, neg_test = random_split(pos_edges,neg_edges,args.split)
    
    print('Positive edges for train: {} and test: {}'.format(len(true_train),len(true_test)))
    print('Negative edges for train: {} and test: {}'.format(len(neg_train),len(neg_test)))
    
    X_train, y_train = model_data(true_train,neg_train,embedding)
    print('Obtained training features!!')
    
    X_test, y_test   = model_data(true_test,neg_test,embedding)
    print('Obtained testing features!!')
    
    model = linear_model.LogisticRegression(random_state=0)
    print("model built!!")
    print("training the model!!")
    model.fit(X_train, y_train)
    print("training complete!!")
    
    pred_y = model.predict(X_test)
    
    test_auc = metrics.roc_auc_score(y_test,pred_y)
    print('AUC of the embedding: {}'.format(test_auc))