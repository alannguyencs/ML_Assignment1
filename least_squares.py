import numpy as np
import matplotlib.pyplot as plt


def least_squares(polynomial_degree=5):
    x = np.loadtxt('./data/polydata_data_sampx.txt')
    y = np.loadtxt('./data/polydata_data_sampy.txt')
    gt_x = np.loadtxt('./data/polydata_data_polyx.txt')
    gt_y = np.loadtxt('./data/polydata_data_polyy.txt')

    #training phase
    Y = y.reshape(-1, 1)
    phi = [x**0]
    for k in range(1, polynomial_degree+1):
        phi = np.append(phi, [x**k], axis=0)

    phi_T = np.transpose(phi)
    product_phi_inverse = np.linalg.inv(np.matmul(phi, phi_T))
    theta_hat = np.matmul(np.matmul(product_phi_inverse, phi), Y)

    #testing phase
    print (theta_hat)
    predicted_phi = [gt_x**0]
    for k in range(1, polynomial_degree+1):
        predicted_phi = np.append(predicted_phi, [gt_x**k], axis=0)
    predicted_phi_T = np.transpose(predicted_phi)
    predicted_Y = np.matmul(predicted_phi_T, theta_hat)

    #visualize data
    plt.plot(x, y, 'ro', label='samples with noises')
    plt.plot(gt_x, gt_y, 'b.', label='ground truth')
    plt.plot(gt_x, predicted_Y, 'gv', label='predicted results')
    plt.legend(loc='upper right', shadow=True, fontsize='x-large')
    plt.show()


least_squares()



