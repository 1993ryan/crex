# coding=utf-8

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
from torchsampler import ImbalancedDatasetSampler

# 固定随机种子、复现结果
seed = 99
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# cpu or GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu1'


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


def autoL1L2(data, norms='l1'):
    '''L1或L2正则化'''
    return preprocessing.normalize(data, norm=norms)

def get_data(feature_file, label_file):
    '''
        1. 获取features特征向量，并正则化
        2. 获取标签labels
        3. 转为np数组返回
    '''
    # 1. features
    features = []
    with open(feature_file, 'r', encoding='utf8') as f:
        for line in f:
            line = eval(line.strip())
            features.append(line)
    # 转为np数组并正则化
    features = np.array(features)
    # features = autoL1L2(features, norms='l2')
    # 2. labels
    labels = []
    with open(label_file, 'r', encoding='utf8') as f:
        for line in f:
            line = eval(line.strip())
            labels.append(line)
    labels = np.array(labels)

    return features, labels


##### 定义网络 #####
class DNN(nn.Module):
    def __init__(self, feature_size=768, hidden_size=32):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(feature_size, hidden_size)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.classifier(x)
        return x

##### 定义dataset #####
class MydataSet(Dataset):
    def __init__(self, features, labels):
        self.x_data = features
        self.y_data = labels
        self.len = len(labels)

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len

    def get_labels(self):
        return self.y_data


##### 训练函数 #####
def log_rmse(net,x,y,criterion):
    net.eval()
    x, y = torch.tensor(x, dtype=torch.float).to(device), torch.tensor(y).long().to(device)
    output = net(x)
    pred = torch.argmax(output,1)

    loss = criterion(output,y)

    true_y, pred = y.tolist(), pred.tolist()
    acc = accuracy_score(true_y, pred)
    precision = precision_score(true_y, pred, average='macro')
    recall = recall_score(true_y, pred, average='macro')
    f1 = f1_score(true_y, pred, average='macro')
    auc = roc_auc_score(true_y, pred)

    net.train()

    return loss.data.item(), acc, (precision, recall, f1, auc)

##### 计算TP #####
def log_tp_tn(net, x,y):
    '''
        FN：False Negative,被判定为负样本，但事实上是正样本。
        FP：False Positive,被判定为正样本，但事实上是负样本。
        TN：True Negative,被判定为负样本，事实上也是负样本。
        TP：True Positive,被判定为正样本，事实上也是证样本。
    '''
    net.eval()
    x, y = torch.tensor(x, dtype=torch.float).to(device), torch.tensor(y).long().to(device)
    output = net(x)
    pred = torch.argmax(output,1)

    # TP：True Positive,被判定为正样本，事实上也是证样本。
    TP_list = ((pred.data == 1) & (y.data == 1)).cpu().long()
    print('预测-TP')
    print(str({'pred_TP': pred.tolist() }))
    print('标签-TP：')
    print(str({'label_TP': y.tolist() }))
    print('TP: ')
    print(str( {'TP_list': TP_list.tolist() } ))

    # TN：True Negative,被判定为负样本，事实上也是负样本。
    TN_list = ((pred.data == 0) & (y.data == 0)).cpu().long()
    print('预测-TN')
    print(str({'pred_TN': pred.tolist() }))
    print('标签-TN：')
    print(str({'label_TN': y.tolist() }))
    print('TN: ')
    print(str( {'TN_list': TN_list.tolist() } ))

    # FP：False Positive,被判定为正样本，但事实上是负样本。
    FP_list = ((pred.data == 1) & (y.data == 0)).cpu().long()

    # FN：False Negative,被判定为负样本，但事实上是正样本。
    FN_list = ((pred.data == 0) & (y.data == 1)).cpu().long()

    with open('log/Labels_{}_TP.csv'.format(type_name), 'w', encoding='utf8') as f:
        f.write('Labels,Labels_{},Index\n'.format(type_name))
        for index, label in enumerate(TP_list):
            if label:
                f.write('1,1,{}\n'.format(str(index)))

    with open('log/Labels_{}_TN.csv'.format(type_name), 'w', encoding='utf8') as f:
        f.write('Labels,Labels_{},Index\n'.format(type_name))
        for index, label in enumerate(TN_list):
            if label:
                f.write('0,0,{}\n'.format(str(index)))

    with open('log/Labels_{}_FP.csv'.format(type_name), 'w', encoding='utf8') as f:
        f.write('Labels,Labels_{},Index\n'.format(type_name))
        for index, label in enumerate(FP_list):
            if label:
                f.write('0,1,{}\n'.format(str(index)))

    with open('log/Labels_{}_FN.csv'.format(type_name), 'w', encoding='utf8') as f:
        f.write('Labels,Labels_{},Index\n'.format(type_name))
        for index, label in enumerate(FN_list):
            if label:
                f.write('1,0,{}\n'.format(str(index)))



def train(net, criterion, optimizer, num_epochs, batch_size, train_features, train_labels, test_features, test_labels):
    train_ls, test_ls = [], []
    dataset = MydataSet(train_features, train_labels)
    train_iter = DataLoader(dataset, batch_size,
                            sampler=ImbalancedDatasetSampler(dataset),
                            # shuffle=True
                            ) #TensorDataset和DataLoader

    for epoch in range(num_epochs):
        ep_train_ls = []
        for X, y in train_iter:
            X, y = X.float().to(device), y.to(device)
            outputs= net(X)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_train_ls.append(loss.item())
        ### 得到每个epoch的 loss 和 accuracy
        t_loss = np.mean(ep_train_ls)
        train_ls.append(t_loss)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print_log = 'Epoch: {}/{}, lr: {}, train_loss: {}'.format(epoch, num_epochs,
                                                                  round(lr, 5),
                                                                  round(t_loss, 4) )
        if test_labels is not None:
            v_loss, v_acc, _ = log_rmse(net, test_features, test_labels, criterion)
            test_ls.append(v_loss)
            print_log += ', test_loss: {}, acc: {}'.format(round(v_loss, 4), v_acc)
        if epoch in [0, 5, 10] or epoch % 20==0:
            print(print_log)
    if test_labels is not None:
        v_loss, v_acc, res = log_rmse(net, test_features, test_labels, criterion)
        print('ACC: {}, P: {}, R: {}, F: {}, AUC: {}'.format(
            v_acc, res[0], res[1], res[2], res[3]
        ))
        return v_acc, res[0], res[1], res[2], res[3]
    else:
        return None


def k_fold_train(features, labels, k=10):
    # 0. 参数
    learning_rate = 0.00001
    weight_decay = 0.9
    num_epochs = 100
    batch_size = 32

    all_acc, all_p, all_r, all_f, all_auc = [], [], [], [],[]
    # k折交叉
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=99)
    i = 1
    feature_size = features.shape[1]
    print('该特征的维度大小为: {}'.format(feature_size))
    for train_index, test_index in kf.split(features, labels):
        # 1. 数据
        train_features, train_labels = features[train_index], labels[train_index]
        test_features, test_labels = features[test_index], labels[test_index]

        # 2. 网络、损失函数、优化器
        net = DNN(feature_size = feature_size)
        criterion = nn.CrossEntropyLoss() ###申明loss函
        #这里使用了Adam优化算法
        optimizer = torch.optim.Adam(params=net.parameters(), lr= learning_rate, weight_decay=weight_decay)
        net = net.to(device)

        # 3. 训练
        print('*'*25,'第','[{}/{}]折'.format(i,k) ,'*'*25)
        _acc, _p, _r, _f, _auc = train(net, criterion, optimizer, num_epochs, batch_size, train_features, train_labels, test_features, test_labels)
        all_acc.append(_acc)
        all_p.append(_p)
        all_r.append(_r)
        all_f.append(_f)
        all_auc.append(_auc)
        # 计算TP
        if i==k:
            log_tp_tn(net, features, labels)
            torch.cuda.empty_cache()
        i += 1
    print('平均：ACC: {}, PRECISION: {}, RECALL: {}, F1: {}, AUC: {}'.format(
        np.mean(all_acc), np.mean(all_p), np.mean(all_r), np.mean(all_f), np.mean(all_auc)
    ))



def main(feature_file, label_file):
    # 1. 准备数据
    features, labels = get_data(feature_file, label_file)

    # 2. 训练
    k_fold_train(features, labels,  k=10)




if __name__ == '__main__':
    # type_list = ['bert', 'cc2vec', 'doc2vec', 'trex', 'word2vec']

    parser = argparse.ArgumentParser(description="Classifier")
    parser.add_argument("--type", default='bert', type=str, help="type name")
    args = parser.parse_args()
    type_name = args.type

    if not os.path.exists('log'):
        os.makedirs('log')

    log_file =  os.path.join('log', type_name+'.txt')
    a = Logger(log_file)
    print('提取特征的网络模型为: {}'.format(type_name))
    sys.stdout = a
    feature_file = os.path.join('data', type_name+'.txt')
    label_file = os.path.join('data', 'lable.txt')
    main(feature_file, label_file)
    a.close()

    '''
        python main.py --type=bert 
        python main.py --type=cc2vec 
        python main.py --type=doc2vec
        python main.py --type=trex 
        python main.py --type=word2vec
    '''
