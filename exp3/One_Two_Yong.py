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


    print('Logistic regression')
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    CLF_LR = LogisticRegression()
    #score = cross_val_score(CLF_LR, Features, Labels, cv=10, scoring='accuracy')
    y_pred=cross_val_predict(CLF_LR, Features, Labels, cv=10)

    print('平均accuracy: ', score.mean())
    print('每则accuracy: ', score)
    score = cross_val_score(CLF_LR, Features, Labels, cv=10, scoring='precision')
    print('平均precision: ', score.mean())
    print('每则precision: ', score)
    score = cross_val_score(CLF_LR, Features, Labels, cv=10, scoring='recall')
    print('平均recall: ', score.mean())
    print('每则recall: ', score)
    score = cross_val_score(CLF_LR, Features, Labels, cv=10, scoring='f1')
    print('平均f1: ', score.mean())
    print('每则f1: ', score)
    score = cross_val_score(CLF_LR, Features, Labels, cv=10, scoring='roc_auc')
    print('平均auc: ', score.mean())
    print('每则auc: ', score)

    ml_name, py_name = 'Logistic regression', 'One_Two_Yong'
    CLF_LR.fit(train_features, train_labels)
    res = CLF_LR.predict(Features)
    pred =  torch.from_numpy(res.reshape(-1,1))
    labels = torch.from_numpy(Labels)
    l = Labels.squeeze()
    write_cross(type_name, pred, labels, res, l, ml_name, py_name)

    # print('Naive bayes')
    # from sklearn.naive_bayes import GaussianNB
    # CLF_NB = GaussianNB()
    # score = cross_val_score(CLF_NB, Features, Labels, cv=10, scoring='accuracy')
    # print('平均accuracy: ', score.mean())
    # print('每则accuracy: ', score)
    # score = cross_val_score(CLF_NB, Features, Labels, cv=10, scoring='precision')
    # print('平均precision: ', score.mean())
    # print('每则precision: ', score)
    # score = cross_val_score(CLF_NB, Features, Labels, cv=10, scoring='recall')
    # print('平均recall: ', score.mean())
    # print('每则recall: ', score)
    # score = cross_val_score(CLF_NB, Features, Labels, cv=10, scoring='f1')
    # print('平均f1: ', score.mean())
    # print('每则f1: ', score)
    # score = cross_val_score(CLF_NB, Features, Labels, cv=10, scoring='roc_auc')
    # print('平均auc: ', score.mean())
    # print('每则auc: ', score)

    # ml_name, py_name = 'Naive bayes', 'One_Two_Yong'
    # CLF_NB.fit(train_features, train_labels)
    # res = CLF_NB.predict(Features)
    # pred =  torch.from_numpy(res.reshape(-1,1))
    # labels = torch.from_numpy(Labels)
    # l = Labels.squeeze()
    # write_cross(type_name, pred, labels, res, l, ml_name, py_name)

    # if not os.path.exists('log'):
    #     os.makedirs('log')
    f_txt = open('{}.txt'.format(type_name), 'w', encoding='utf8')
    TP_list = ((pred.data == 1) & (labels.data == 1)).long()
    f_txt.write(str({'pred_TP': res}))
    f_txt.write('\n')
    f_txt.write(str({'label_TP': l}))
    f_txt.write('\n')
    f_txt.write(str({'TP_list': TP_list.squeeze().tolist()}))
    f_txt.write('\n')

    # TN：True Negative,被判定为负样本，事实上也是负样本。
    TN_list = ((pred.data == 0) & (labels.data == 0)).long()
    f_txt.write(str({'TN_list': TN_list.squeeze().tolist()}))
    f_txt.write('\n')

    # FP：False Positive,被判定为正样本，但事实上是负样本。
    FP_list = ((pred.data == 1) & (labels.data == 0)).long()

    # FN：False Negative,被判定为负样本，但事实上是正样本。
    FN_list = ((pred.data == 0) & (labels.data == 1)).long()



    with open('Labels_{}_TP.csv'.format(type_name), 'w', encoding='utf8') as f:
        f.write('Labels,Labels_{},Index\n'.format(type_name))
        for index, label in enumerate(TP_list):
            if label:
                f.write('1,1,{}\n'.format(str(index)))

    with open('Labels_{}_TN.csv'.format(type_name), 'w', encoding='utf8') as f:
        f.write('Labels,Labels_{},Index\n'.format(type_name))
        for index, label in enumerate(TN_list):
            if label:
                f.write('0,0,{}\n'.format(str(index)))

    with open('Labels_{}_FP.csv'.format(type_name), 'w', encoding='utf8') as f:
        f.write('Labels,Labels_{},Index\n'.format(type_name))
        for index, label in enumerate(FP_list):
            if label:
                f.write('0,1,{}\n'.format(str(index)))

    with open('Labels_{}_FN.csv'.format(type_name), 'w', encoding='utf8') as f:
        f.write('Labels,Labels_{},Index\n'.format(type_name))
        for index, label in enumerate(FN_list):
            if label:
                f.write('1,0,{}\n'.format(str(index)))
