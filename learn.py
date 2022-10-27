# import numpy as np
from mxnet import autograd,nd

#自动求梯度
""" 
#生成4*1的矩阵
x = nd.arange(4).reshape((4,1))
print(x)
x.attach_grad()         ##申请储存梯度所需要的内存
with autograd.record():
    y = 2 * nd.dot(x.T,x)             #这里的x是标量，所以y也是标量

y.backward()   ###调用backward函数自动求梯度，如果y不是一个标量，mxnet将默认先对y中元素求和得到新的变量，再求该变量有关x的梯度
# assert (x.grad -4*x).norm().asscalar() == 0
print(x.grad)

print(autograd.is_training())
with autograd.record():
    print(autograd.is_training())
"""

### 对python控制流求梯度

