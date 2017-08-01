import theano
from theano import tensor
# declare two symblic floating-point scalar
a = tensor.dscalar()
b = tensor.dscalar()
# create a simple expression
c = a + b
# convert the expression into calculatable object that take (a,b)
# values as input and compute a value for c
f = theano.function([a,b], c)
# bind 1.5 to a, 2.5 to b and evaluate c
result = f(1.5, 2.5)
print(result)
