import numpy as np
import matplotlib.pyplot as plt



def compute_parameters(polynomial_degree=5, alpha=0.88):
    x = np.loadtxt('./data/polydata_data_sampx.txt')
    y = np.loadtxt('./data/polydata_data_sampy.txt')

    dimension = polynomial_degree + 1


    # training phase
    Y = y.reshape(-1, 1)                                #N x 1
    phi = [x ** 0]                                      #k x N
    for k in range(1, polynomial_degree + 1):
        phi = np.append(phi, [x ** k], axis=0)

    
    #compute sum_
    phi_T = np.transpose(phi)
    phi_phi_T = np.matmul(phi, phi_T)
    I_d = np.identity(dimension)
    sum_hat = np.linalg.inv(1./alpha*I_d + 1./(alpha**2)*phi_phi_T)

    #compute mean_hat
    mean_hat = 1./(alpha**2) * np.matmul(sum_hat, np.matmul(phi, Y))

    return mean_hat, sum_hat




def visualize_results(mean_hat, sum_hat, polynomial_degree=5):
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

    mean_star = np.matmul(predicted_phi_T, mean_hat)
    sum_star = np.matmul(predicted_phi_T, np.matmul(sum_hat, predicted_phi))

    print (mean_star.shape)
    print (sum_star.shape)

    predicted_Y = mean_star

    # visualize data
    plt.plot(x, y, 'ro', label='samples with noises')
    plt.plot(gt_x, gt_y, 'b.', label='ground truth')
    plt.plot(gt_x, predicted_Y, 'gv', label='predicted results')
    plt.legend(loc='upper right', shadow=True, fontsize='x-large')
    plt.show()




mean_hat, sum_hat = compute_parameters()
visualize_results(mean_hat=mean_hat, sum_hat=sum_hat)
