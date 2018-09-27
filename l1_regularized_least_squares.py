import numpy as np
import matplotlib.pyplot as plt


def compute_theta(polynomial_degree=5, _lambda=10):
    x = np.loadtxt('./data/polydata_data_sampx.txt')
    y = np.loadtxt('./data/polydata_data_sampy.txt')

    # training phase
    Y = y.reshape(-1, 1)                                #N x 1
    phi = [x ** 0]                                      #k x N
    for k in range(1, polynomial_degree + 1):
        phi = np.append(phi, [x ** k], axis=0)

    lambda_I =  _lambda * np.identity(polynomial_degree+1)
    phi_T = np.transpose(phi)
    product_phi_inverse = np.linalg.inv(np.matmul(phi, phi_T) + lambda_I)
    theta_hat = np.matmul(np.matmul(product_phi_inverse, phi), Y)

    print (theta_hat)
    np.save('./results/l1_regularized_least_squares.npy', theta_hat)


def visualize_results():
    theta_hat = np.load('./results/l1_regularized_least_squares.npy')
    print (theta_hat)

compute_theta()
visualize_results()



