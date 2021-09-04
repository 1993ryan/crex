import os
import glob

# comapare different files, return line num
def compare_file(file1,file2):
    f1 = open(file1,'r')
    f2 = open(file2,'r')
    count = 0
    diff = []
    for line1 in f1:
        line2 = f2.readline()
        count +=1
        if line1 != line2:
            diff.append(count)
    f1.close()
    f2.close()
    return diff


def absoulte_path(path, fileAbsolutePath):
    # dict = {'name': 'runoob', 'likes': 123, 'url': 'www.runoob.com'}
    valid_arch_emb = []
    valid_byte1 = []
    valid_byte2 = []
    valid_byte3 = []
    valid_byte4 = []
    valid_inst_pos_emb = []
    valid_op_pos_emb = []
    valid_static = []

    file_list = os.listdir(path)
    for file in file_list:
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            absoulte_path(cur_path, fileAbsolutePath)
        else:
            val = cur_path.split('/')[-1]
            print(val)

            if 'valid.arch_emb' in val:
                valid_arch_emb.append(cur_path)
            if 'valid.byte1' in val:
                valid_byte1.append(cur_path)
            if 'valid.byte2' in val:
                valid_byte2.append(cur_path)
            if 'valid.byte3' in val:
                valid_byte3.append(cur_path)
            if 'valid.byte4' in val:
                valid_byte4.append(cur_path)
            if 'valid.inst_pos_emb' in val:
                valid_inst_pos_emb.append(cur_path)
            if 'valid.op_pos_emb' in val:
                valid_op_pos_emb.append(cur_path)
            if 'valid.static' in val:
                valid_static.append(cur_path)
            

            # fileAbsolutePath.append(cur_path)

            break

    return valid_arch_emb, valid_byte1, valid_byte2, valid_byte3, valid_byte4, valid_inst_pos_emb, valid_op_pos_emb, valid_static
    # return fileAbsolutePath

if __name__ == '__main__':
    
    fileAbsolutePath = {}

    valid_arch_emb, valid_byte1, valid_byte2, valid_byte3, valid_byte4, valid_inst_pos_emb, valid_op_pos_emb, valid_static = absoulte_path('/home/dataset', fileAbsolutePath)

    all_file = [valid_arch_emb, valid_byte1, valid_byte2, valid_byte3, valid_byte4, valid_inst_pos_emb, valid_op_pos_emb, valid_static]

    for file in all_file:
        print(file)
    # file_coconut = []
    # file_gt = []
    # file_bugcode = []
    
    # for i, file in enumerate(file_list):
    #     if i % 3 == 0:
    #         file_coconut.append(file)
    #     if i % 3 == 1:
    #         file_gt.append(file)
    #     if i % 3 == 2:
    #         file_bugcode.append(file)
        
    # print('coconut', file_coconut)
    # print('/n')
    # print('gt', file_gt)
    # print('/n')
    # print('bugcode', file_bugcode)

    # for coconut, gt, bugcode in zip(file_coconut, file_gt, file_bugcode):
    #     print(coconut)
    #     print(gt)
    #     print(bugcode)
    #     bugcode_vs_coconut = compare_file(bugcode, coconut)
    #     bugcode_vs_gt = compare_file(bugcode, gt)
    #     print('bugcode_vs_coconut:', bugcode_vs_coconut)
    #     print('bugcode_vs_gt:', bugcode_vs_gt)


    # Yan
    # extract_path = '/home/trex/tools/output/'
    # extract_out_path = '/home/trex/tools/extract_out/'

    # hang = 3

    # files = os.listdir(extract_path)
    # try:
    #     for file in files:
    #         print(file)
    #         with open( extract_path + file ,'r') as f:
    #             data = f.readlines()
    #         print(data[hang-1])
    #         row_data = data[hang - 1]
    #         if '\n' in row_data:
    #             pass
    #         else:
    #             row_data += '\n'
    #         with open(extract_out_path + file.replace('.input1','').replace('.input0',''),'a+') as f:
    #             f.write(row_data)
    # except:
    #     print('index out')


    # file1 = input('输入第一个文件：')
    # file2 = input('输入第二个文件：')
    # file1 = '/home/dataset/620-B-bug-15846277-15846335/bugcode/valid.arch_emb.620-B-15846277'
    # file2 = '/home/dataset/620-B-bug-15846277-15846335/Coconut/valid.arch_emb.620-B-15846277-CoCoNut'
    # differ = compare_file(file1,file2)
    # print(differ)
    # if len(differ) == 0:
    #     print('两个文件一样')
    # else:
    #     print('两个文件共有%d处不同' % len(differ))
    #     for each in differ:
    #         print('第%d行不同' % each)