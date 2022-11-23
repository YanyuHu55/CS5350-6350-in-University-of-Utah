import numpy as np
from scipy.optimize import minimize
import math

# kernel
def gaussian_kernel(train_x, gamma):
    G_Kernel = np.zeros((train_x.shape[0],train_x.shape[0]))
    for i in range(0,train_x.shape[0]):
        for j in range(0,train_x.shape[0]):
            X = train_x[i] - train_x[j]
            G_Kernel[i,j] = math.exp(-np.linalg.norm(X, ord=2)/gamma)
    # print(G_Kernel)
    # print(G_Kernel.shape[0]) #872
    # print(G_Kernel.shape[1]) #872
    return G_Kernel

def fun(alpha, xy):

    return 0.5*np.matmul(np.matmul(alpha,xy),np.transpose(alpha)) - np.sum(alpha)

def jac(alpha,xy):

    return np.matmul(alpha,xy)-np.ones(np.transpose(alpha).shape[0])


def dual_gaussian(train_data, C, gamma):
    train_x = train_data[:, 0:-1]  # train_x.shape[0] = 872, train_x.shape[1] = 4

    # print(train_x)
    train_y = train_data[:, -1]
    train_y[train_y == 0] = -1
    num_data = train_x.shape[0]
    # print(num_data)
    x0 = np.random.uniform(0, C, size=num_data)
    x0_transpose = np.reshape(x0, (-1, 1))
    # print(x0_transpose)
    # x0 = np.random.rand(num_data)
    # print(x0)

    train_y = np.reshape(train_y, (-1, 1))

    # matrix
    yy_t = np.dot(train_y, np.transpose(train_y))
    # print(yy_t)
    # print(yy_t.shape[0]) #872
    # print(yy_t.shape[1])#872
    kernel = gaussian_kernel(train_x, gamma)
    xy = np.multiply(yy_t, kernel)
    # print(xy)
    bnds = [(0, C)] * num_data
    # print(bnds)

    # fun = lambda alpha: 0.5*np.matmul(np.matmul(alpha,xy),np.transpose(alpha)) - np.sum(alpha)
    cons = [{'type': 'eq', 'fun': lambda alpha: np.sum(np.multiply(alpha, np.transpose(train_y)))}]
    # #optimal
    optimal = minimize(fun, x0, args=(xy,), method='SLSQP', jac=jac, bounds=bnds, constraints=cons)
    # print(optimal)
    # print(optimal.x.shape[0])
    # w,b
    w = np.sum(np.multiply(np.multiply(optimal.x, np.transpose(train_y)), np.transpose(train_x)), axis=1)
    b = np.mean(np.transpose(train_y) - np.sum(np.matmul(np.multiply(optimal.x,np.transpose(train_y) ), np.transpose(kernel)),axis=0))
    # b = np.mean(np.transpose(train_y) - np.matmul(w, np.transpose(train_x)))
    # print(w)
    # print(b)
    alpha_optimal = optimal.x
    return w, b, alpha_optimal

def errorRate(test_data, b, alpha_optimal,gamma):
    test_y = test_data[:, -1]
    test_x = test_data[:, 0:-1]
    test_y[test_y == 0] = -1
    kernel = gaussian_kernel(test_x, gamma)
    # print(alpha_optimal)
    # print(alpha_optimal.shape[0])
    # print(alpha_optimal.shape[1])
    alpha = alpha_optimal[0:test_x.shape[0]]

    error = 0.0
    w_x = np.matmul(np.multiply(alpha,np.transpose(test_y) ), np.transpose(kernel)) + b
    # print(w_x)
    # print(w_x.shape[0])
    w_x[w_x > 0] = 1
    w_x[w_x <= 0] = -1
    # print(w_x)
    test_y = np.reshape(test_y, (-1, 1))
    w_x = np.reshape(w_x,(-1,1))
    for i in range(0, w_x.shape[0]):
        if w_x[i] != test_y[i]:
            error += 1
    error_rate = error / test_x.shape[0]
    # print('error rate:', error_rate)
    return error_rate

train_data = np.loadtxt('train.csv', dtype='float32', delimiter=",")
test_data = np.loadtxt('test.csv', dtype='float32', delimiter=",")
gamma = [0.1, 0.5, 1, 5, 100]
C = 500/873
for i in range(0, len(gamma)-1):
    w1, b1, alpha_optimal1 = dual_gaussian(train_data, C, gamma[i])
    w2, b2, alpha_optimal2 = dual_gaussian(train_data, C, gamma[i+1])
    alpha_optimal1 = alpha_optimal1[alpha_optimal1 != 0]
    # print(alpha_optimal1)
    # print(alpha_optimal1.shape[0])
    alpha_optimal2 = alpha_optimal2[alpha_optimal2 != 0]
    # print(alpha_optimal2)
    # print(alpha_optimal2.shape[0])
    intersection = np.intersect1d(alpha_optimal1,alpha_optimal2)
    print(intersection)
    inter_num = intersection.shape[0]
    print('the number of overlapped support vector between gamma = ',gamma[i],' and gamma = ', gamma[i+1], 'is:')
    print(inter_num)

