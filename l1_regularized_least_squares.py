import numpy as np
# import matplotlib.pyplot as plt
from qpsolvers import solve_qp


def compute_theta(polynomial_degree=1, _lambda=10):
    x = np.loadtxt('./data/polydata_data_sampx.txt')
    y = np.loadtxt('./data/polydata_data_sampy.txt')

    dimension = polynomial_degree + 1
    Dimension = 2 * dimension

    # training phase
    Y = y.reshape(-1, 1)                                #N x 1
    phi = [x ** 0]                                      #k x N
    for k in range(1, polynomial_degree + 1):
        phi = np.append(phi, [x ** k], axis=0)

    

    phi_T = np.transpose(phi)
    phi_phi_T = np.matmul(phi, phi_T)
    print (phi_phi_T.shape)
    print (phi_phi_T)


    H = np.bmat([[phi_phi_T, -phi_phi_T], [-phi_phi_T, phi_phi_T]])
    H = np.asarray(H)
    print (H)

    lambda_one = (_lambda * np.ones(Dimension)).reshape(-1, 1)
    phi_Y = np.matmul(phi, Y)
    f = np.asarray(lambda_one - np.bmat([[phi_Y], [-phi_Y]]))
    
    print (type(f))
    print (f.shape)

    f = f.reshape((Dimension,))
    # print (f[:5])
    # print (f.shape)

    G = np.identity(Dimension)

    print (type(H), H.shape)
    
    h = np.zeros(Dimension)

    

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




    print ("H:", type(H), H.shape)
    print ("f:", type(f), f.shape)
    print ("G:", type(G), G.shape)
    print ("h:", type(h), h.shape)
    print ("QP solution:", solve_qp(H, f, G, h))
    

    # print (theta_hat)
    # np.save('./results/l1_regularized_least_squares.npy', theta_hat)


def visualize_results():
    theta_hat = np.load('./results/l1_regularized_least_squares.npy')
    print (theta_hat)

compute_theta()
# visualize_results()



