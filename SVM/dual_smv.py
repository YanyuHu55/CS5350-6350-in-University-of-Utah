import numpy as np
from scipy.optimize import minimize

# function
def fun(alpha, xy):

    return 0.5*np.matmul(np.matmul(alpha,xy),np.transpose(alpha)) - np.sum(alpha)

def jac(alpha,xy):

    return np.matmul(alpha,xy)-np.ones(np.transpose(alpha).shape[0])



def dual_smv(train_data, C):
    train_x = train_data[:, 0:-1]  # train_x.shape[0] = 872, train_x.shape[1] = 4

    # print(train_x)
    train_y = train_data[:, -1]
    train_y[train_y == 0] = -1
    num_data = train_x.shape[0]
    # print(num_data)
    x0 = np.random.uniform(0, C , size=num_data)
    x0_transpose = np.reshape(x0 , (-1, 1))
    # print(x0_transpose)
    # x0 = np.random.rand(num_data)
    # print(x0)

    train_y = np.reshape(train_y , (-1, 1))

    yy_t = np.dot(train_y,np.transpose(train_y))

    xx_t = np.dot(train_x,np.transpose(train_x))

    xy = np.multiply(yy_t , xx_t)

    bnds = [(0,C)]* num_data
    # print(bnds)

    # fun = lambda alpha: 0.5*np.matmul(np.matmul(alpha,xy),np.transpose(alpha)) - np.sum(alpha)
    cons = [{'type':'eq','fun':lambda alpha : np.sum(np.multiply(alpha, np.transpose(train_y)))}]
    # #optimal
    optimal = minimize(fun, x0, args=(xy,), method='SLSQP', jac=jac, bounds=bnds, constraints = cons)
    # print(optimal)
    # print(optimal.x.shape[0])
    #w,b
    w = np.sum(np.multiply(np.multiply(optimal.x,np.transpose(train_y)),np.transpose(train_x)),axis=1)
    b = np.mean(np.transpose(train_y)-np.matmul(w,np.transpose(train_x)))
    # print(w)
    # print(b)
    return w,b

def errorRate(test_data, w, b):
    test_y = test_data[:, -1]
    test_x = test_data[:, 0:-1]
    test_y[test_y == 0] = -1
    error = 0.0
    w_x = np.matmul(w, np.transpose(test_x)) + b
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
C = [100/873, 500/873, 700/873]
for i in range(0, len(C)):
    w,b = dual_smv(train_data,C[i])
    # w = [-0.94288663, -0.65150745, -0.73357073, -0.04122781]
    # b = 2.516855921540314
    print('when C is ', C[i])
    print('w is: ',w)
    print('b is: ',b)
    error_rate = errorRate(test_data, w, b)
    print('error rate is: ', error_rate)
