import os
import linecache

def absoulte_path(path, fileAbsolutePath):
    file_list = os.listdir(path)
    for file in file_list:
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            absoulte_path(cur_path, fileAbsolutePath)
        else:
            val = cur_path.split('/')[-1]
            # print(val)
            if 'valid' in val:
                fileAbsolutePath.append(cur_path)
    
    return fileAbsolutePath

if __name__ == '__main__':
    fileAbsolutePath = []
    file_list = absoulte_path('/test/bc_gt', fileAbsolutePath)
    print(len(file_list))
    # for i, files in enumerate(file_list):
    #     # dir, file = os.path.split(files)
    #     # print(dir)
    #     if (i+1) % 8 == 0 and i != 0:
    #         print(files)
    #         print(i)
    #         with open(files, encoding='utf-8') as f:
    #             data = linecache.getline(files,4)
    #             print(data)
    #             with open(extract_out_path + file.replace('.input1','').replace('.input0',''),'a+') as f:
    #             f.write(row_data)

    for i, files in enumerate(file_list):

        dir, file = os.path.split(files)
        modify_path = dir + '/' + 'modify/'
        # print(modify_path)
        # print(file)
        with open(files, encoding='utf-8') as f:
            data = linecache.getline(files,4)
            # print(data)

        with open(modify_path + file.replace('.funtion',''),'w') as f:    #设置文件对象
            f.write(data)                 #将字符串写入文件中
