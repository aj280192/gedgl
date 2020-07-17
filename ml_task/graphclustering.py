import numpy as np
import argparse
import logging

def data_preprocessing(emb_file,label_file):
    data, labels = {}, {}
    with open(emb_file) as e:
        for line in e.readlines()[1:]: #skipping the first line of embedding file
            values = line.strip().split(' ')
            data[int(values[0])] = list(map(float,values[1:]))
    
    with open(label_file) as l:
        for line in l.readlines():
            node, label = line.strip().split(' ')
            labels[int(node)] = int(label)
    
    X = np.array(list(data.values()))
    y = []
    for node in data.keys():
        y.append(labels[node])
    y = np.array(y)
    
    return X,y

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--embfile", type=str, required=True,
            help="embedding file path")
    parser.add_argument("--labelfile", type=str, required=True,
            help="label file path")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(message)s')
    
    X, y = data_preprocessing(args.embfile,args.labelfile)

from sklearn.cluster import KMeans
from sklearn import metrics

num_clust = [2,4,8,10,16,20,30,32,40]
print('Will be calcualting NMI for cluster numbers: ' + str(num_clust))

scores = []

for clusters in num_clust:
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)
    c_labels = kmeans.labels_
    score = metrics.normalized_mutual_info_score(y,c_labels)
    scores.append(round(score,6))

print('The NMI scores are :' + str(scores))