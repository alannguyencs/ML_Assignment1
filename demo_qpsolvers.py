import numpy as np
from qpsolvers import solve_qp

M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
P = np.dot(M.T, M)  # quick way to build a symmetric matrix
q = np.dot(np.array([3., 2., 3.]), M).reshape((3,))
# G = array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
G = np.identity(3)
# print (P)
# print (type(P))
# print (P.shape)
h = np.zeros(3)

print ("P:", type(P), P.shape)
print ("q:", type(q), q.shape)
print ("G:", type(G), G.shape)
print ("h:", type(h), h.shape)

print ("QP solution:", solve_qp(P, q, G, h))



#ref: https://github.com/stephane-caron/qpsolvers