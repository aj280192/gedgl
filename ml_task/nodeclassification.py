import numpy as np
import argparse
import logging
from sklearn import model_selection 
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier


def data_preprocessing_mc(emb_file,label_file):
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

def data_preprocessing_ml(emb_file, label_file):
    data, labels = {}, {}
    with open(emb_file) as e:
        for line in e.readlines()[1:]: #skipping the first line of embedding file
            values = line.strip().split(' ')
            data[int(values[0])] = list(map(float,values[1:]))
    
    with open(label_file) as l:
        for line in l.readlines():
            row = line.strip().split(' ')
            labels[int(row[0])] = list(map(int,row[1:]))
    
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
    parser.add_argument("--type", type=int, required=True,
            help="1. - multi-class 2. - multi-label")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(message)s')

if args.type == 1:
    
    X, y = data_preprocessing_mc(args.emb_file,args.label_file)
    
    print('creating linear model for multi-class classification!!')
    model = linear_model.LogisticRegression(C=10, random_state=0, multi_class = 'multinomial', max_iter=500)

else:
    
    X, y = data_preprocessing_ml(args.emb_file,args.label_file)
    
    print('creating one-vs-rest with linear model for multi-label classification!!')
    model = linear_model.LogisticRegression(C=10, random_state=0, multi_class = 'multinomial', max_iter=500)
    model = OneVsRestClassifier()


scoring = ['f1_macro','f1_micro']

print('performing CV for the model!!')
cv_scores = model_selection.cross_validate(model,X,y,scoring=scoring,cv=5,return_train_score=True)

train_micro_f1 = cv_scores['train_f1_micro'].mean()
test_micro_f1 = cv_scores['test_f1_micro'].mean()
train_macro_f1 = cv_scores['train_f1_macro'].mean()
test_macro_f1 = cv_scores['test_f1_macro'].mean()

print("Train acc: f1_micro {:0.4f}, f1_macro: {:0.4f}".format(train_micro_f1, train_macro_f1))
print("Test acc: f1_micro {:0.4f}, f1_macro: {:0.4f}".format(test_micro_f1, test_macro_f1))