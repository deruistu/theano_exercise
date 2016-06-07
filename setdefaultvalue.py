'''
Created on Feb 24, 2016

@author: lqy
'''
import theano
import theano.tensor as T
from theano import In
from theano import function

x,y = T.dscalars("x","y")

out = x+y

f = function([x, In(y, value=1)],out)

print f(2,3)