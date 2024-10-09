from pyope.ope import OPE


def encrypt(cipher, data):
    res = []
    for i in range(len(data)):
        res.append(cipher.encrypt(int(data[i])))
    return res


def decrypt(cipher, data):
    res = []
    for i in range(len(data)):
        res.append(cipher.decrypt(int(data[i])))
    return res
