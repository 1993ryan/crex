import numpy as np
import sys
import pandas as pd
pd.set_option('display.max_rows', None) # 展示所有行
pd.set_option('display.max_columns', None) # 展示所有列
import sklearn
import torch
import warnings
warnings.filterwarnings('ignore')
import os
import time
import random
random.seed(99)
from utils import  write_cross
train_index = random.sample(list(range(212)) , int(212*0.8))
if not os.path.exists('log/One_Two_Yong'):
    os.makedirs('log/One_Two_Yong')

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def close(self):
        self.log.close()

log_file = 'log/One_Two_Yong/res.txt'
a = Logger(log_file)
sys.stdout = a

# for Name in ['bert', 'cc2vec', 'doc2vec', 'trex', 'word2vec']:
for Name in ['bert',  'trex', 'word2vec']:
    train_features = []
    train_labels = []

    type_name = Name
    print('Name: ', Name)
    Features = pd.read_csv('traindata_F_' + Name + '.csv')
    Features['Features'] = Features['Features'].apply(lambda x: x[1:-1].split(', ')).apply(lambda x: [float(y) for y in x])
    Features = pd.DataFrame(list(Features['Features']))
    # old label
    # Labels = pd.read_csv('traindata_L.csv')
    # Labels = np.array(Labels['Labels']).reshape(-1, 1)
    # print(Labels)
    # print(Labels.shape)

    # 新的标签
    Labels = []
    with open('lable_new.txt', 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                Labels.append([int(line)])
    Labels = np.array(Labels)
    # print(Labels)
    # print(Labels.shape)
    for _index in train_index:
        train_labels.append(Labels[_index])
        train_features.append(np.array(Features.iloc[_index, :]))
    train_labels = np.array(train_labels)
    train_features = np.array(train_features)


    # print('DecisionTree')
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    # CLF_DT = DecisionTreeClassifier()
    # score = cross_val_score(CLF_DT, Features, Labels, cv=10, scoring='accuracy')
    # print('平均accuracy: ', score.mean())
    # print('每则accuracy: ', score)
    # score = cross_val_score(CLF_DT, Features, Labels, cv=10, scoring='precision')
    # print('平均precision: ', score.mean())
    # print('每则precision: ', score)
    # score = cross_val_score(CLF_DT, Features, Labels, cv=10, scoring='recall')
    # print('平均recall: ', score.mean())
    # print('每则recall: ', score)
    # score = cross_val_score(CLF_DT, Features, Labels, cv=10, scoring='f1')
    # print('平均f1: ', score.mean())
    # print('每则f1: ', score)
    # score = cross_val_score(CLF_DT, Features, Labels, cv=10, scoring='roc_auc')
    # print('平均auc: ', score.mean())
    # print('每则auc: ', score)
    #
    # ml_name, py_name = 'DecisionTree', 'One_Two_Yong'
    # CLF_DT.fit(train_features, train_labels)
    # res = CLF_DT.predict(Features)
    # pred =  torch.from_numpy(res.reshape(-1,1))
    # labels = torch.from_numpy(Labels)
    # l = Labels.squeeze()
    # write_cross(type_name, pred, labels, res, l, ml_name, py_name)


    print('Random Forest')
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    CLF_RF = RandomForestClassifier()
    #score = cross_val_score(CLF_LR, Features, Labels, cv=10, scoring='accuracy')
    y_pred=cross_val_predict(CLF_RF, Features, Labels, cv=10)

    from sklearn.metrics import *
    tn,fp,fn,tp=confusion_matrix(Labels,y_pred).ravel()
    # recall1=tp/(tp+fn)
    recall=recall_score(Labels,y_pred)
    acc=accuracy_score(Labels,y_pred)
    prec=precision_score(Labels,y_pred)
    f1=f1_score(Labels,y_pred)
    auc=cross_val_score(CLF_RF, Features, Labels, cv=10, scoring='roc_auc').mean()
    # print(recall1)
    print(recall)
    print(acc)
    print(prec)
    print(f1)
    print(auc)


    df = pd.DataFrame(Features)
    df["label"]=Labels
    df["y_pred"]=y_pred
    masktp=(df["label"]==1) & (df["y_pred"]==1)
    masktn = (df["label"] == 0) & (df["y_pred"] == 0)
    maskfp = (df["label"] == 0) & (df["y_pred"] == 1)
    maskfn = (df["label"] == 1) & (df["y_pred"] == 0)
    df["tp"]=masktp
    df["tn"] = masktn
    df["fp"] = maskfp
    df["fn"] = maskfn
    df.to_csv("./rf/"+Name+"202109031111.结果.csv")

    pd.DataFrame(df.loc[masktp,:].index).to_csv("./rf/"+Name+"tp.csv")
    pd.DataFrame(df.loc[masktn, :].index).to_csv("./rf/"+Name + "tn.csv")
    pd.DataFrame(df.loc[maskfp, :].index).to_csv("./rf/"+Name + "fp.csv")
    pd.DataFrame(df.loc[maskfn, :].index).to_csv("./rf/"+Name + "fn.csv")

