import glob
import json
import math
import re
import subprocess
import time
from itertools import product

from capstone import *
import random
import string

import subprocess
from collections import defaultdict
from itertools import product

from capstone import *
import json

from tracing import EX
import shutil
import glob


import os.path


def objdump(file):
    result = subprocess.run(['objdump', '-d', file, '-j', '.text'], capture_output=True, text=True, check=True)
    return result.stdout


# input: the .text part of the objdump's output for one elf file
# output:the dictionary, function_name:bytes delimited by space
def parse_objdump_output(s):
    funcname = ''
    ret_func_dict = {}

    for line in s.split('\n'):
        # function name, and this is the start
        funcname_re = re.findall(r'\<(.*)\>:', line)
        if len(funcname_re) == 1:
            funcname = funcname_re[0]
            ret_func_dict[funcname] = []
            continue

        elif len(funcname_re) > 1:
            print(f'error, more than one functions matched: {funcname_re}')
            exit()

        # match bytes
        bytes_re = re.findall(r':\t(.*?)\t', line)
        if len(bytes_re) == 1:
            inst_bytes = bytes_re[0].strip()
            ret_func_dict[funcname].extend(inst_bytes.split())


        elif len(bytes_re) == 0:
            # handle nop line
            bytes_re_nop = re.findall(r':\t(.*?)\s+$', line)
            if len(bytes_re_nop) == 1:
                inst_bytes = bytes_re_nop[0].strip()
                ret_func_dict[funcname].extend(inst_bytes.split())

        elif len(bytes_re) > 1:
            print(f'error, more than one byte strings matched: {bytes_re}')
            exit()

    return ret_func_dict


def str2byte(s):
    return bytes(bytearray.fromhex(s))


def tokenize(s):
    s = s.replace(',', ' , ')
    s = s.replace('[', ' [ ')
    s = s.replace(']', ' ] ')
    s = s.replace(':', ' : ')
    s = s.replace('*', ' * ')
    s = s.replace('(', ' ( ')
    s = s.replace(')', ' ) ')
    s = s.replace('{', ' { ')
    s = s.replace('}', ' } ')
    s = s.replace('#', '')
    s = s.replace('$', '')
    s = s.replace('!', ' ! ')

    s = re.sub(r'-(0[xX][0-9a-fA-F]+)', r'- \1', s)
    s = re.sub(r'-([0-9a-fA-F]+)', r'- \1', s)

    # s = re.sub(r'0[xX][0-9a-fA-F]+', 'hexvar', s)

    return s.split()


def byte2instruction(s, md):
    ret_tokens = []

    try:
        for _, _, op_code, op_str in md.disasm_lite(s, 0x400000):
            tokens = tokenize(f'{op_code} {op_str}')
            for token in tokens:
                ret_tokens.append(token)
    except CsError as e:
        print("ERROR: %s" % e)

    return ' '.join(ret_tokens)


def num_inst(s, md):
    return len(list(md.disasm_lite(s, 0x400000)))


def hex2str(s):
    num = s.replace('0x', '')
    assert len(num) <= 8
    num = '0' * (8 - len(num)) + num
    return num


def get_trace(static_codes, registers_list):
    code = []
    trace = []
    inst_indices = []
    token_indices = []
    for inst_index, (inst, registers) in enumerate(zip(static_codes, registers_list)):
        tokens = tokenize(inst)

        for token_index, token in enumerate(tokens):
            if token.upper() in registers:
                code.append(token.lower())
                trace.append(hex2str(hex(registers[token.upper()])))
            elif '0x' in token.lower():
                code.append('hexvar')
                trace.append(hex2str(token.lower()))

            elif token.lower().isdigit():
                code.append('num')
                trace.append(hex2str(token.lower()))
            else:
                code.append(token.lower())
                trace.append('#' * 8)
            inst_indices.append(inst_index)
            token_indices.append(token_index)

    return code, trace, inst_indices, token_indices


def absoulte_path(path, fileAbsolutePath):
    file_list = os.listdir(path)
    for file in file_list:
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            absoulte_path(cur_path, fileAbsolutePath)
        else:
            dir, s_filename = os.path.split(cur_path)
            
            if glob.glob(os.path.join(dir, '*.c')):
                if '.c' in s_filename:
                    fileAbsolutePath.append(os.path.join(dir, s_filename))
        
    return fileAbsolutePath


if __name__ == '__main__':
    path = []
    fileAbsolutePath = absoulte_path('/test/bc_gt', path)
    # filename = fileAbsolutePath[0]
    # print(filename.split('/')[-1].split('.c.c')[0])
    for filenames in fileAbsolutePath:
        dir, file01 = os.path.split(filenames)
        filename = file01.split('/')[-1].split('.c')[0]
        dir1 = dir + '/'+ filename

        # if not os.path.exists(dir1):
        #     # print(filename)
        #     delfile = dir1.split('/')[0]+'/'+dir1.split('/')[1]+'/'+dir1.split('/')[2]+'/'+dir1.split('/')[3]
        #     print(delfile)
        #     # os.remove(delfile)
        #     shutil.rmtree(delfile)
        #     # os.removedirs(delfile)
            
        
        # filename = 'input1'
        # path = '/home/trex/tools/{}'.format(filename)
        # file_list = [path]
        # filename = 'input1'
        # path = '/home/trex/tools/{}'.format(filename)


        file_list = [dir1]
        func_dict = {}
        type = filename

        # print(filenames)

        # if compiled file exists, break
        if not os.path.exists(dir+'/valid.static.{}'.format(type)):
            print(dir+'/valid.static.{}'.format(type))
            
            
            mds = {'x86-32': Cs(CS_ARCH_X86, CS_MODE_32),
                'x86-64': Cs(CS_ARCH_X86, CS_MODE_64),
                'arm-32': Cs(CS_ARCH_ARM, CS_MODE_ARM + CS_MODE_LITTLE_ENDIAN),
                'arm-64': Cs(CS_ARCH_ARM64, CS_MODE_ARM + CS_MODE_LITTLE_ENDIAN),
                'mips-32': Cs(CS_ARCH_MIPS, CS_MODE_MIPS32 + CS_MODE_BIG_ENDIAN),
                'mips-64': Cs(CS_ARCH_MIPS, CS_MODE_MIPS64 + CS_MODE_BIG_ENDIAN)}
            md = mds['x86-64']
            num_traces = 1
            timeout = 50000

            for i, file in enumerate(file_list):
                objdump_result = objdump(file)
                ret_func_dict = parse_objdump_output(objdump_result)
                for funcname, funcbody in ret_func_dict.items():
                    if funcname in func_dict:
                        func_dict[funcname][f'{file}'] = funcbody
                    else:
                        func_dict[funcname] = {f'{file}': funcbody}
            if func_dict:
                with open('/home/trex/tools/objdump/{}.json'.format(type),'w') as f:
                    json.dump(func_dict,f)
                # print(func_dict)
                functraces = {}
                for i, funcname in enumerate(func_dict):
                    if funcname.startswith('_'):
                        continue


                    for j, binfile in enumerate(func_dict[funcname]):

                        # if binfile != 'data-raw/bin/arm-32/coreutils-8.26-O0/sha384sum':
                        #     continue

                        # funcbody is list of bytes, we covert bytes to instructions
                        code = str2byte(' '.join(func_dict[funcname][binfile]))

                        # skip functions that are too short
                        if num_inst(code, md) < 5:
                            continue

                        # print(binfile, '...')
                        for _ in range(num_traces):
                            # create emulator for dynamic tracing micro execution
                            ex = EX(code, mode='x86', md=md)

                            try:
                                ex.run(timeout=timeout)
                            except Exception as e:
                                print(f"{e}: {binfile} {funcname}")

                            instruction_body, traces, inst_indices, token_indices = get_trace(ex.mu.static, ex.mu.trace)


                            if funcname in functraces and binfile in functraces[funcname]:
                                functraces[funcname][f'{binfile}'].append(
                                    (instruction_body, traces, inst_indices, token_indices))
                            elif funcname in functraces:
                                functraces[funcname][f'{binfile}'] = [
                                    (instruction_body, traces, inst_indices, token_indices)]
                            else:
                                functraces[funcname] = {
                                    f'{binfile}': [(instruction_body, traces, inst_indices, token_indices)]}

                        # dummy trace to enforce model to focus on static prediction
                        traces_dummy = ['#' * 8] * len(traces)
                        functraces[funcname][f'{binfile}'].append(
                            (instruction_body, traces_dummy, inst_indices, token_indices))
                print(functraces)
                instruction_body_list = []
                traces_list = []
                traces1_list = []
                traces2_list = []
                traces3_list = []
                traces4_list = []
                inst_indices_list = []
                token_indices_list = []

                x86_list = []
                for i in functraces.values():
                    data = i.values()
                    for i in data:
                        for j in i:
                            instruction_body, traces, inst_indices, token_indices  = j
                            data_len = len(instruction_body)
                            traces = traces[:data_len]
                            inst_indices = inst_indices[:data_len]
                            token_indices = token_indices[:data_len]
                            instruction_body_list.append(' '.join(instruction_body))
                            x86_list.append(' '.join(['x86'] * len(instruction_body)))

                            traces_list.append(' '.join(traces))
                            traces1_list.append(' '.join([i[:2] for i in traces]))
                            traces2_list.append(' '.join([i[2:4] for i in traces]))
                            traces3_list.append(' '.join([i[4:6] for i in traces]))
                            traces4_list.append(' '.join([i[-2:] for i in traces]))

                            token_indices_list.append(' '.join(str(i) for i in token_indices))

                            inst_pos_data = token_indices
                            inst_pos_data.append(0)
                            start = 0
                            inst_pos = []
                            last_zero = 0
                            for i, elm in enumerate(inst_pos_data):
                                if elm == 0 and i != 0:
                                    flag = i
                                    inst_pos.extend([start] * (flag - last_zero))
                                    last_zero = flag
                                    start += 1
                            inst_indices_list.append(' '.join([str(i) for i in inst_pos]))


                with open(dir+'/valid.static.{}'.format(type),'w') as f:
                    f.writelines('\n'.join(instruction_body_list[::2]))
                with open(dir+'/valid.byte1.{}'.format(type),'w') as f:
                    f.writelines('\n'.join(traces1_list[::2]))
                with open(dir+'/valid.byte2.{}'.format(type),'w') as f:
                    f.writelines('\n'.join(traces2_list[::2]))
                with open(dir+'/valid.byte3.{}'.format(type),'w') as f:
                    f.writelines('\n'.join(traces3_list[::2]))
                with open(dir+'/valid.byte4.{}'.format(type),'w') as f:
                    f.writelines('\n'.join(traces4_list[::2]))
                print(inst_indices_list)
                with open(dir+'/valid.inst_pos_emb.{}'.format(type),'w') as f:
                    f.writelines('\n'.join(inst_indices_list[::2]))
                with open(dir+'/valid.op_pos_emb.{}'.format(type),'w') as f:
                    f.writelines('\n'.join(token_indices_list[::2]))
                with open(dir+'/valid.arch_emb.{}'.format(type),'w') as f:
                    f.writelines('\n'.join(x86_list[::2]))
                

                # with open(dir+'/valid.static.{}'.format(type),'w') as f:
                #     f.writelines('\n'.join(instruction_body_list[::2]))
                # with open(dir+'/valid.byte1.{}'.format(type),'w') as f:
                #     f.writelines('\n'.join(traces1_list[::2]))
                # with open(dir+'/valid.byte2.{}'.format(type),'w') as f:
                #     f.writelines('\n'.join(traces2_list[::2]))
                # with open(dir+'/valid.byte3.{}'.format(type),'w') as f:
                #     f.writelines('\n'.join(traces3_list[::2]))
                # with open(dir+'/valid.byte4.{}'.format(type),'w') as f:
                #     f.writelines('\n'.join(traces4_list[::2]))
                # print(inst_indices_list)
                # with open(dir+'/valid.inst_pos_emb.{}'.format(type),'w') as f:
                #     f.writelines('\n'.join(inst_indices_list[::2]))
                # with open(dir+'/valid.op_pos_emb.{}'.format(type),'w') as f:
                #     f.writelines('\n'.join(token_indices_list[::2]))
                # with open(dir+'/valid.arch_emb.{}'.format(type),'w') as f:
                #     f.writelines('\n'.join(x86_list[::2]))

            

