import copy
import random

import matplotlib.pyplot as plt

import math
import numpy as np

def stochasticGD(maxIteration, r, train_x, train_y):
    w = [0.0 for _ in range(0, train_x.shape[1])]
    # w=[2,1,1,1,1,1,1]
    w = list(w)
    times = 0

    w = np.array(w)
    # print(w)
    d = train_x.shape[1]
    m = train_x.shape[0]

    cost = []
    while times < maxIteration:
        times += 1

        i = random.randrange(0, m)
        j = 0
        # print('i = ', i)
        error = train_y[i] - np.matmul(w, train_x[i])
        # print('error: ', error)
        for j in range(0, d):
            gradient = r * error * train_x[i, j]
            # print('gradient = ', gradient)
            w_new = w[j] + gradient
            # print('w new: ', w_new)
            w[j] = w_new
            # print('the jth of w: ', w[j])
        # print('w', w)
        sub_cost = 0
        for num in range(0, m):
            # print(num, 'th y is: ', train_y[num])
            # print(num, 'th X is: ', train_x[num])

            num_cost = np.square(train_y[num] - np.dot(w, train_x[num]))
            sub_cost = sub_cost + num_cost

        sub_cost = sub_cost * 0.5
        cost.append(copy.deepcopy(sub_cost))
        # print('cost value is', cost)

    return w, cost



train_data = np.loadtxt('train.csv', dtype='float32', delimiter=",")

train_y = train_data[:,-1]

train_x = train_data[:,0:-1]


r = 0.001
maxIteration = 10000
w, cost_train = stochasticGD(maxIteration, r, train_x, train_y)
print('w', w)
print('cost value is', cost_train)

plt.figure()
plt.plot(cost_train, color='green', linewidth=2)
plt.xlabel('iteration times')
plt.ylabel('cost')
plt.xlim((-100,10000))
plt.ylim((14,25))
plt.show()


# test
test_data = np.loadtxt('test.csv', dtype='float32', delimiter=",")
test_y = test_data[:,-1]
test_x = test_data[:,0:-1]

# test cost
m = test_x.shape[0]
sum_square = 0.0
# print('m: ', m)
for i in range(0,m):
    error = test_y[i] - np.matmul(w, test_x[i])
    error_square = np.square(error)
    sum_square = sum_square + error_square
cost_test = 0.5 * sum_square
print('the cost value of test is: ', cost_test)
print('weight vector is ', w)
