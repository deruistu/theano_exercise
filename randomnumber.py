'''
Created on Feb 24, 2016

@author: lqy
'''
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams 

srng = RandomStreams(seed = 234) 
rv_u = srng.uniform((1,1))
rv_n = srng.normal((1,1))
f = theano.function([],rv_u)
g = theano.function([],rv_n, no_default_updates=True)
#nearly_zeros = theano.function([],rv_u+rv_u-2*rv_u)
print (f())
print (f())
print (g())
print (g())
print (rv_u)
print (rv_n)