import copy
import matplotlib.pyplot as plt
import numpy as np



def Batch(w, train_x, train_y, maxIteration):
    # w=[2,1,1,1,1,1,1]
    w = list(w)
    times = 0
    w = np.array(w)
    # print(w)
    convergence = 1.0
    d = train_x.shape[1]
    m = train_x.shape[0]

    r = 0.0001
    lost =[]
    while convergence > 1e-6 and times < maxIteration:
        times += 1
        gradient = []
        lostForj = []
        for j in range(0, d):
            sum_errorlist = []
            sub_lost = []
            for i in range(0, m):
                # print('train i :', train_x[i])
                # print('train y :', train_y[i])
                newtrain_x = np.transpose(train_x[i])

                error = train_y[i] - np.matmul(w, train_x[i])
                # print(error)
                input_error = error * train_x[i, j]
                # print(input_error)
                errorSquare = np.square(error)
                # print('error square:', errorSquare)
                sum_errorlist.append(copy.deepcopy(input_error))
                # print('sum_error list :', sum_errorlist)
                sub_lost.append(copy.deepcopy(errorSquare))
                # print('sub lost: ', sub_lost)
            grad_elem = -np.sum(sum_errorlist)
            # print('gradient sum: ', grad_elem)
            lost_elem = np.sum(sub_lost)
            lostForj.append(copy.deepcopy(lost_elem))
            # print('lost element: ', lost_elem)
            gradient.append(copy.deepcopy(grad_elem))
            # print('gradient', gradient)
        # w_new1 = [r * i for i in gradient]
        w_new = w - [r * i for i in gradient]
        # print('w new1\n', w_new1)
        # print('w new\n', w_new)
        difference = w_new - w
        convergence = np.linalg.norm(difference)
        # print('convergence: ', convergence)
        w = w_new
        sum_lost = 0.5 /d * np.sum(lostForj)
        lost.append(copy.deepcopy(sum_lost))

    # print('2final w is', w)
    # print('Loss is', lost)
    return w, lost


# train_data = pd.read_csv('train.csv', header=None)
# train_data.head()
train_data = np.loadtxt('train.csv', dtype='float32', delimiter=",")
# print('numpy\n',train_data)
train_y = train_data[:,-1]
# train_y1 = list(train_y)
# print('y = \n', train_y)
train_x = train_data[:,0:-1]
# train_x1 = train_x.tolist()
# print('x = \n', train_x)
maxIteration = 3000
w_test = [0.0 for _ in range(0, train_x.shape[1])]
w, lost = Batch(w_test, train_x, train_y, maxIteration)
print('final w is', w)

plt.figure()
plt.plot(lost, color='green', linewidth=2)
plt.xlabel('iteration times')
plt.ylabel('cost')
plt.xlim((0,3000))
plt.ylim((14,25))
plt.show()
# test
test_data = np.loadtxt('test.csv', dtype='float32', delimiter=",")
test_y = test_data[:,-1]
test_x = test_data[:,0:-1]
#
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


#
# w_test, cost_test = Batch(test_x, test_y, maxIteration)
# print('w', w_test)
# print('cost value is', cost_test)
#
# plt.figure()
# plt.plot(cost_test, color='red', linewidth=2)
# plt.xlabel('iteration times')
# plt.ylabel('cost')
# plt.xlim((-100,3000))
# plt.ylim((14,30))
# plt.show()








