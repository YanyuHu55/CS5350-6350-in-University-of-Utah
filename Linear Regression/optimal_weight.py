
import numpy as np


train_data = np.loadtxt('train.csv', dtype='float32', delimiter=",")
train_y = train_data[:,-1]
train_x = train_data[:,0:-1]
train_x_transpose = np.transpose(train_x)

xx_t = np.matmul(train_x_transpose, train_x)
# print('XX^T = ', xx_t)
inverse =np.linalg.inv(xx_t)
# print('inverse = ', inverse)
inverse_x = np.matmul(inverse, train_x_transpose)
# print('inverse * x', inverse_x)
optimal_w = np.matmul(inverse_x, train_y)
print('optimal weight vector is: ', optimal_w)



