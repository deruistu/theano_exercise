'''
Created on Feb 24, 2016

@author: lqy
'''
#注意第14行和15行的变量定义和引用
from __future__ import print_function
import theano
a = theano.tensor.vector()  # declare variable
b = theano.tensor.vector()  # declare variable
out = a ** 2 + b ** 2 + 2 * a * b  # build symbolic expression
f = theano.function([a, b], out)   # compile function
print(f([1, 2], [4, 5]))  # prints [ 25.  49.]
