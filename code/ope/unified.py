# 实验统一接口
from random import random, randint

import numpy as np
from pyope.ope import OPE, ValueRange

from script.ope.BCLO import BCLO
from script.ope.GACD import GACD
from script.ope.CCS16 import POPE_list_encryption


class CipherOPE(object):
    def __init__(self, type, key):
        self.type = type
        self.key = key
        # FHOPE计数
        if type == 'FHOPE':
            # self.tag = 0
            # self.ciphertexts_state = []
            self.plaintexts = []
            self.count = []
        if type == 'BCLO':
            self.cipher = OPE(key, in_range=ValueRange(0, 100000000), out_range=ValueRange(10000000000, 1000000000000))

    # 参数：datas是一组向量
    def uni_batch(self, datas):
        res = []
        if self.type == 'BCLO':
            for i in datas:
                res.append(BCLO.encrypt(self.cipher, i))
        elif self.type == 'GACD':
            for i in datas:
                res.append(GACD.encrypt(self.key, i))
        elif self.type == 'POPE':
            for i in datas:
                ct = POPE_list_encryption.encrypt_(i)
                s = sorted(i)
                temp = []
                for m in range(len(s)):
                    temp.append([s[m], ct[m]])
                res_t = []
                for n in i:
                    for q in temp:
                        if q[0] == n:
                            res_t.append(q[1])
                res.append(res_t)
        elif self.type == 'FHOPE':
            for i in datas:
                for j in i:
                    if j in self.plaintexts:
                        self.count[self.plaintexts.index(j)] = self.count[self.plaintexts.index(j)] + 1
                    else:
                        self.plaintexts.append(j)
                        self.count.append(1)
            state_temp = sorted(zip(self.plaintexts, self.count), key=lambda x: x[0])
            for i in range(len(state_temp)):
                state_temp[i] = list(state_temp[i])
            for i in range(len(state_temp)):
                if i == 0:
                    continue
                state_temp[i][1] = state_temp[i][1] + state_temp[i - 1][1]
            for i in datas:
                res_col = []
                for j in i:
                    for x in range(len(state_temp)):
                        if j == state_temp[x][0]:
                            if x == 0:
                                res_col.append(randint(0, state_temp[x][1]))
                            else:
                                res_col.append(randint(state_temp[x - 1][1], state_temp[x][1]))
                            break
                res.append(res_col)

            # for i in datas:
            #     temp = Li_encryption.encrypt_(self.tag, i, self.ciphertexts_state)
            #     self.tag = 1
            #     self.ciphertexts_state = temp
            # temp_arr = np.transpose(np.array(self.ciphertexts_state), (1, 0)).tolist()
            #
            # for data in datas:
            #     res_t = []
            #     for item in data:
            #         # 从floor到ceil取样
            #         res_t.append(temp_arr[1][temp_arr[0].index(item)+randint(1, temp_arr[0].count(item))-1])
            #     res.append(res_t)

        return res


# 密钥设置
# OPE配置
key_ope = OPE.generate_key()
# GACD配置
M = 10000
key_GACD = GACD.generate_key(M)

bclo = CipherOPE('BCLO', key_ope)
gacd = CipherOPE('GACD', key_GACD)
pope = CipherOPE('POPE', 'key')
fhope = CipherOPE('FHOPE', 'key')
# print(bclo.uni_batch([[1, 3, 2], [0, 5525, 3849], [0, 5525, 3849]]))
# print(gacd.uni_batch([[1, 3, 2], [0, 5525, 3849], [0, 5525, 3849]]))
# print(pope.uni_batch([[1, 3, 2], [0, 5525, 3849], [0, 5525, 3849]]))
# print(fhope.uni_batch([[1], [0], [9]]))
#
