import os

import os

def multifilesRead(path, filesAbsolutePath):
    file_list = os.listdir(path);
    for file in file_list:
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            multifilesRead(cur_path, filesAbsolutePath)
        else:
            filesAbsolutePath.append(cur_path)
    return filesAbsolutePath

filesAbsolutePath = []
print(multifilesRead('/home/repair'), filesAbsolutePath)