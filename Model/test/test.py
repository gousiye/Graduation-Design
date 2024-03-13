# import numpy as np

# a = np.array([[1,2,3],[4,5,6]])
# b = np.array([[7,8,9],[10,11,12]])

# print(np.concatenate((a,b),axis=1))

# class Test:
#     def __init__(self):
#         self.a = 12

# def test():
#     a = Test()
#     print(id(a))
#     return a


# b = test()
# print(id(b))

# name = "Alice"
# score = 99

# a = "name is {name}, score is {score}".format(name = name, score = score)
# print(a)

# class Test:
#     def __init__(self, dictA):
#         for key in dictA:
#             print(key)


# myDict = {'a' : 1, 'b' : 2, 'c' : 3}
# test = Test(myDict)
# import torch
# a = torch.tensor([1, 2, 3], dtype=torch.float32)
# b = torch.tensor([4, 5, 6], dtype=torch.float32)
# c = torch.tensor([7, 8, 100], dtype=torch.float32)
# d = torch.stack((a,b,c))
# print(d.mean())

# import datetime


# now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
# print(now_time)

import torch

# 假设我们有一个Tensor x
x = torch.tensor([1.0, 2.0, 3.0])

# 将x中的所有元素设置为0
x.fill_(0)
print(x[0:3])