import numpy as np
import random
import argparse
import logging
from sklearn import linear_model
from sklearn import metrics

def data_preprocessing(emb_file,edge_list):
    data = {}
    with open(emb_file) as e:
        for line in e.readlines()[1:]: #skipping the first line of embedding file
            values = line.strip().split(' ')
            data[int(values[0])] = list(map(float,values[1:]))
    
    nodes = list(data.key())
            
    print("embedding value loaded!!")
    
    edges = []
    with open(edge_list) as l:
        for line in l.readlines():
            node, label = line.strip().split(' ')
            edges.append((int(node),int(label)))
    
    print("edge list created!!")
    
    return data, nodes, edges

def random_split(edges,split):
    
    total = len(edges)
    split_index = int(total/(100*split))    
    random.shuffle(edges)
    train = edges[:split_index]
    test = edges[split_index:]
    return train,test

def add_negative(nodes,edges):
    """ defaulted to add 30% negatives """
    
    neg_count = int(len(edges) * .3)
    node_len = len(nodes)

    neg_edges = []
    
    for _ in range(neg_count):
        trail = 0
        while True:
            x = nodes[random.randint(0,node_len-1)]
            y = nodes[random.randint(0,node_len-1)]
            trail += 1
            if trail >= 1000:
                all_flag = True
                break
            if x != y and (x,y) not in edges and (y,x) not in edges:
                neg_edges.append((x,y))
                break
        if all_flag:
            break
    
    return neg_edges

def model_data(true_edge,neg_edge,embedding):
    
    true_data = []
    for n1, n2 in true_edge:
        node1 = np.array(embedding[n1])
        node2 = np.array(embedding[n2])
        edge_data = np.multiply(node1,node2)
        true_data.append(edge_data)
    
    true_y = np.ones(len(true_data))
    
    neg_data = []
    for n1, n2 in neg_edge:
        node1 = np.array(embedding[n1])
        node2 = np.array(embedding[n2])
        edge_data = np.multiply(node1,node2)
        neg_data.append(edge_data)
    
    neg_y = np.zeros(len(neg_data))
    
    X = np.concatenate(np.array(true_data),np.array(neg_data))
    y = np.concatenate(true_y,neg_y)
    
    return X,y

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--embfile", type=str, required=True,
            help="embedding file path")
    parser.add_argument("--edgefile", type=str, required=True,
            help="edge list file path")
    parser.add_argument("--datasplit", type=float, default=0.3,
            help="train and test split")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(message)s')
    
    embedding, nodes, edges = data_preprocessing(args.emb_file,args.label_file)
    print('Obtained nodes and edgelist!!')
    true_edge_train, true_edge_test = random_split(edges,args.datasplit)
    print('edgelist split into train and test')
    
    neg_edge_train = add_negative(nodes,true_edge_train)
    neg_edge_test  = add_negative(nodes,true_edge_test)
    print('adding negative to the edges!!')
    
    print('Positive edges for train: {} and test: {}'.format(len(true_edge_train),len(true_edge_test)))
    print('Negative edges for train: {} and test: {}'.format(len(neg_edge_train),len(neg_edge_test)))
    
    X_train, y_train = model_data(true_edge_train,neg_edge_train,embedding)
    print('Obtained training features!!')
    
    X_test, y_test   = model_data(true_edge_test,neg_edge_test,embedding)
    print('Obtained testing features!!')
    
    model = linear_model.LogisticRegression(random_state=0)
    print("model built!!")
    print("training the model!!")
    model.fit(X_train, y_train)
    print("training complete!!")
    
    pred_y = model.predict_proba(X_test)
    
    test_auc = metrics.roc_auc_score(y_test,pred_y)
    print('AUC of the embedding: {}'.format(test_auc))