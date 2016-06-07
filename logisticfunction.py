'''
Created on Feb 24, 2016

@author: lqy
'''
import theano
import theano.tensor as T


x = T.dmatrix("x")  ## first, define the type of the input matrix
y = T.dmatrix("y")
#y = 1/(1+T.exp(-x)) ## define the expression Formula
out = x+y 

f = theano.function([x,y],out)  # define the calculation function

print (f([[1,2,3],[4,5,6]],[[1,1,1],[1,1,1]]))
