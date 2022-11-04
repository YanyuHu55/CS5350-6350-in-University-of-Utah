import numpy as np

# train_data = np.loadtxt('train.csv', dtype='float32', delimiter=",")
# print(train_data)
# print(train_data.shape[0]) # the number of row : 872
# print(train_data.shape[1]) # the number of column : 5
def standard_per(train_data):
    m = 0
    n=0
    r=0.001
    w = np.zeros(train_data.shape[1], dtype=float)
    w=np.reshape(w, (-1,train_data.shape[1])) # w.shape[0] = 1, w.shape[1] = 5, row = 1, column = 4
    # print(w)
    # print(w.shape[0])
    # print(w.shape[1])
    for t in range(0,10):

        np.random.shuffle(train_data)
        #print(train_data)
        train_x = np.copy(train_data) # train_x.shape[0] = 872, train_x.shape[1] = 5
        train_x[:, -1] = 1
        train_y = train_data[:, -1] # train_y.shape[0] = 872
        # trans_x = np.transpose(train_x[0])
        # print(trans_x)
        train_y[train_y == 0] =-1
        # print(train_data.shape[0])
        # print(train_y)
        for i in range(0, train_data.shape[0]): #train_x.shape[0]
            # print('w= ',w)
            update_sign = train_y[i]*np.matmul(w, np.transpose(train_x[i]))
            # print(update_sign)
            m += 1
            if update_sign <= 0:
                w_new = w + r*train_y[i]*train_x[i]
                # print(w_new)
                w = w_new
                n += 1
                # print(w.shape[0])
                # print(w.shape[1])
            # print(update_sign.shape[0])
            # print(update_sign.shape[1])
    print('final:', w)
    # print('m:',m)
    # print('n:',n)
    return w


def error(test_data, w):
    test_y = test_data[:, -1]
    test_x = np.copy(test_data)
    test_x[:, -1] = 1
    test_y[test_y == 0] = -1
    error = 0.0

    for i in range(0, test_data.shape[0]):
        update_sign = test_y[i] * np.matmul(w, np.transpose(test_x[i]))
        if update_sign <= 0:
            error += 1
    error_rate = error / test_data.shape[0]
    print('error rate: ', error_rate)
    return error_rate


train_data = np.loadtxt('train.csv', dtype='float32', delimiter=",")
w = standard_per(train_data)
print(w)

test_data = np.loadtxt('test.csv', dtype='float32', delimiter=",")
# print('test data: ',test_data)
# print(test_data.shape[0])
# print(test_data.shape[1])
error_rate = error(test_data, w)
print('error rate:', error_rate)
