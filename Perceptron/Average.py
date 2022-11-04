import numpy as np



def average_per(train_data):
    m = 0
    n=0
    r=0.001
    w = np.zeros(train_data.shape[1], dtype=float)
    w=np.reshape(w, (-1,train_data.shape[1])) # w.shape[0] = 1, w.shape[1] = 5, row = 1, column = 5
    # print(w)
    # print(w.shape[0])
    # print(w.shape[1])
    a =  np.zeros(train_data.shape[1], dtype=float)
    a = np.reshape(w, (-1,train_data.shape[1]))
    for t in range(0,10):

        np.random.shuffle(train_data)
        #print(train_data)
        train_x = np.copy(train_data)  # train_x.shape[0] = 872, train_x.shape[1] = 4
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
                a = a + w
                # print(w.shape[0])
                # print(w.shape[1])
            # print(update_sign.shape[0])
            # print(update_sign.shape[1])
    print('final:', a)
    # print('m:',m)
    # print('n:',n)
    return a

def error(test_data, a):
    error = 0.0
    test_y = test_data[:, -1]
    test_x = np.copy(test_data)
    test_x[:, -1] = 1
    test_y[test_y == 0] = -1
    a_x = np.matmul(a, np.transpose(test_x))
    # print('a_x: ', a_x)
    # print(a_x.shape[0])
    # print(a_x.shape[1])
    a_x[a_x > 0] = 1
    a_x[a_x <= 0] = -1
    # print('a_x: ', a_x)
    test_y = np.reshape(test_y, (-1, 1))
    a_x = np.reshape(a_x,(-1,1))
    for i in range(0, a_x.shape[0]):
        if a_x[i] != test_y[i]:
            error += 1
    error_rate = error / test_x.shape[0]
    print('error rate:', error_rate)
    return error_rate

train_data = np.loadtxt('train.csv', dtype='float32', delimiter=",")
a = average_per(train_data)

test_data = np.loadtxt('test.csv', dtype='float32', delimiter=",")
error_rate = error(test_data, a)
