# coding=utf-8

# 计算TP
import os
import sys
import numpy as np
from main import Logger

type_list = ['bert', 'cc2vec', 'doc2vec' , 'word2vec']

f_res = open('log/result.csv', 'w', encoding='utf8')
f_res.write('Classifier,Emb,Acc,Prec,Recall,F1,AUC\n')

log_file = 'log/cross.txt'
a = Logger(log_file)
sys.stdout = a

# trex 标签
label_TP_list = []
label_TN_list = []
with open('log/trex.txt', 'r', encoding='utf8') as f:
    for line in f:
        line = line.strip()
        if 'TP_list' in line:
            label_TP_list = eval(line)['TP_list']
            label_TP_list = np.array(label_TP_list)
        if 'TN_list' in line:
            label_TN_list = eval(line)['TN_list']
            label_TN_list = np.array(label_TN_list)
        if '平均' in line:
            _acc, _p, _r, _f, _auc = line.split(',')
            _acc, _p, _r, _f, _auc = _acc.split()[-1], _p.split()[-1], _r.split()[-1], _f.split()[-1], _auc.split()[-1]
            f_res.write(',trex,{},{},{},{},{}\n'.format(str(_acc), str(_p),
                                                        str(_r), str(_f), str(_auc)
                                                        ))

# 其他模型
for type in type_list:
    log_file = os.path.join('log', type+'.txt')
    with open(log_file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if 'TP_list' in line:
                TP_list = eval(line)['TP_list']
                TP_list = np.array(TP_list)
            if 'TN_list' in line:
                TN_list = eval(line)['TN_list']
                TN_list = np.array(TN_list)
            if '平均' in line:
                _acc, _p, _r, _f, _auc = line.split(',')
                _acc, _p, _r, _f, _auc = _acc.split()[-1], _p.split()[-1], _r.split()[-1], _f.split()[-1], _auc.split()[-1]
                f_res.write(',{},{},{},{},{},{}\n'.format(type, str(_acc), str(_p),
                                                            str(_r), str(_f), str(_auc)
                                                            ))

    print('{} 特征'.format(type))
    # 计算TP
    res_P = label_TP_list & TP_list
    print('TP: Total: trex: {}, {}: {}, cross: {}'.format(label_TP_list.sum(),
                                                          type, TP_list.sum(),
                                                          res_P.sum() ))

    # 计算TN
    res_N = label_TN_list & TN_list
    print('TN: Total: trex: {}, {}: {}, cross: {}'.format(label_TN_list.sum(),
                                                          type, TN_list.sum(),
                                                          res_N.sum() ))

f_res.close()
a.close()