'''
Created on Feb 24, 2016

@author: lqy
'''
import theano 
import theano.tensor as T

a,b = T.dmatrices("a","b")

diff = a-b 
abs_diff = abs(diff)
sqr_diff = diff**2

f = theano.function([a,b],[diff,abs_diff,sqr_diff]) # the order of the output objects does not influence any thing. it can be arbitrary

#f = theano.function([a,b],[abs_diff,diff,sqr_diff])

diff_output, abs_diff_output, sqr_diff_output = f([[1,2,3]],[[2,2,2]])

print diff_output
print abs_diff_output
print sqr_diff_output


