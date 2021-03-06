import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog



def compute_theta(polynomial_degree=5):
    x = np.loadtxt('./data/polydata_data_sampx.txt')
    y = np.loadtxt('./data/polydata_data_sampy.txt')

    dimension = polynomial_degree + 1


    # training phase
    Y = y.reshape(-1, 1)                                #N x 1
    phi = [x ** 0]                                      #k x N
    for k in range(1, polynomial_degree + 1):
        phi = np.append(phi, [x ** k], axis=0)

    

    phi_T = np.transpose(phi)
    N = phi_T.shape[0]
    In = np.identity(N)


    A = np.bmat([[-phi_T, -In], [phi_T, -In]])
    print (A[0])
    # A = np.asarray(A)
    print (A.shape)

    b = np.bmat([[-Y], [Y]])
    b = (np.asarray(b)).reshape((2*N,))
    print (b)
    print (b.shape)



    Od = np.zeros(dimension)
    one_n = np.asarray([1 for _ in range(N)])
    f = np.bmat([[Od], [one_n]])
    f = (np.asarray(f)).reshape(((dimension+N),))
    # f_T = np.transpose(f)
    # print (f_T.shape)
    print (f)
    print (f.shape)


    res = linprog(c=f, A_ub=A, b_ub=b, method='simplex')
    print (res)
    print (res['x'])

    theta_hat = np.asarray(res['x'][:dimension])
    print (theta_hat)


    np.save('./results/robust_regression.npy', theta_hat)


def visualize_results(polynomial_degree=5):
    theta_hat = np.load('./results/robust_regression.npy')
    print(theta_hat)

    x = np.loadtxt('./data/polydata_data_sampx.txt')
    y = np.loadtxt('./data/polydata_data_sampy.txt')
    gt_x = np.loadtxt('./data/polydata_data_polyx.txt')
    gt_y = np.loadtxt('./data/polydata_data_polyy.txt')

    predicted_phi = [gt_x ** 0]
    for k in range(1, polynomial_degree + 1):
        predicted_phi = np.append(predicted_phi, [gt_x ** k], axis=0)
    predicted_phi_T = np.transpose(predicted_phi)
    predicted_Y = np.matmul(predicted_phi_T, theta_hat)

    # visualize data
    plt.plot(x, y, 'ro', label='samples with noises')
    plt.plot(gt_x, gt_y, 'b.', label='ground truth')
    plt.plot(gt_x, predicted_Y, 'gv', label='predicted results')
    plt.legend(loc='upper right', shadow=True, fontsize='x-large')
    plt.show()




# compute_theta()
visualize_results()


#ref: https://docs.scipy.org/doc/scipy/reference/optimize.linprog-simplex.html
#     http://yetanothermathprogrammingconsultant.blogspot.com/2017/12/scipy-10-linear-programming.html  

