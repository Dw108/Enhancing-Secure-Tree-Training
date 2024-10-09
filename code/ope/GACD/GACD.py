import numpy as np
import math
import random

# M表示明文范围, K的范围要大于M的8/3次方
M = 10000


# 密钥生成
def generate_key(M):
    _lambda = math.ceil(np.log2(M) * 4)
    k_ceil = 2 ** (_lambda + 1)
    k_floor = 2 ** _lambda
    key = random.randint(k_floor, k_ceil)
    return key


# 加密
def encrypt(key, data):
    res = []
    for i in range(len(data)):
        r = random.randint(math.ceil(key ** (3 / 4)), math.ceil(key - key ** (3 / 4)))
        res.append(int(data[i]) * key + r)
    return res


# 解密
def decrypt(key, data):
    res = []
    for i in range(len(data)):
        res.append(math.floor(int(data[i]) / key))
    return res
