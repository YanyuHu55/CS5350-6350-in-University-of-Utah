import numpy as np



def smv(train_data, num_data, C):
    learningRate = 0.001
    learningRate_0 = 0.001
    a = 0.01
    m = 0 # iteration number
    w = np.zeros(train_data.shape[1], dtype=float)
    # print(w)
    w=np.reshape(w, (-1,train_data.shape[1])) # w.shape[0] = 1, w.shape[1] = 5, row = 1, column = 5
    # print(w)

    for t in range(0,100):
        np.random.shuffle(train_data)
        train_x = np.copy(train_data)  # train_x.shape[0] = 872, train_x.shape[1] = 5
        train_x[:, -1] = 1
        # print(train_x)
        train_y = train_data[:, -1]
        train_y[train_y == 0] = -1
        # print(train_y)
        for i in range(0, train_data.shape[0]): #train_x.shape[0]
            m = m + 1
            update_sign = train_y[i]*np.matmul(w, np.transpose(train_x[i]))
            # print(update_sign)
            if update_sign <= 1:
                w = w - learningRate * w + learningRate * C * num_data * train_y[i] * train_x[i]
                # print(w)
            else:
                # w_0 = w[:,:-1]
                # b_0 = w[:,-1]
                w = (1-learningRate)*w
                # w = np.append(w_0[0],b_0)
                # w = np.reshape(w, (-1,train_data.shape[1]))
                # print(b_0)
                # print(w_0)
            # print(w)

            learningRate = learningRate_0 / (1+m)
            # print(learningRate)
    # print(w)
    return w

def error_rate(w, train_data):
    train_x = np.copy(train_data)  # train_x.shape[0] = 872, train_x.shape[1] = 5
    train_x[:, -1] = 1
    # print(train_x)
    # print(w)
    # print(train_x)
    train_y = train_data[:, -1]
    train_y[train_y == 0] = -1
    error = 0.0
    w_x = np.matmul(w, np.transpose(train_x))
    # print(w_x)
    # print(w_x.shape[0])
    # print(w_x.shape[1])
    w_x[w_x>0] = 1
    w_x[w_x<=0] = -1
    # print(w_x)
    train_y = np.reshape(train_y , (-1, 1))
    w_x = np.reshape(w_x,(-1,1))
    # print(w_x)
    # print(train_y)
    for i in range(0, w_x.shape[0]):
        if w_x[i] != train_y[i]:
            error += 1
    error_rate = error / train_x.shape[0]
    # print('error rate is :',error_rate)
    return error_rate



train_data = np.loadtxt('train.csv', dtype='float32', delimiter=",")

test_data = np.loadtxt('test.csv', dtype='float32', delimiter=",")

# print(train_data)
num_data = train_data.shape[0]
# print(num_data)

C = [100/873, 500/873, 700/873]
for i in range(0,3):
    w = smv(train_data, num_data, C[i])

    train_error = error_rate(w ,train_data)
    test_error = error_rate(w,test_data)
    print(i+1, '. the C is:', C[i])
    print('w = ', w)
    print('train error rate is: ', train_error)
    print('test error rate is: ', test_error)