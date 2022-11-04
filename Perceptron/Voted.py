import numpy as np

# train_data = np.loadtxt('train.csv', dtype='float32', delimiter=",")
def voted_per(train_data):
    m=0
    c = 0
    r = 0.001
    w = np.zeros(train_data.shape[1], dtype=float)
    w = np.reshape(w, (-1, train_data.shape[1]))  # w.shape[0] = 1, w.shape[1] = 5, row = 1, column = 5
    w_array = np.array([])
    c_array = np.array([])

    # print(w)
    # print(w.shape[0])
    # print(w.shape[1])
    for t in range(0,10):
        np.random.shuffle(train_data)
        # print(train_data)
        train_x = np.copy(train_data)  # train_x.shape[0] = 872, train_x.shape[1] = 4
        train_x[:, -1] = 1
        train_y = train_data[:, -1]  # train_y.shape[0] = 872
        train_y[train_y == 0] =-1

        for i in range(0, train_data.shape[0]):  #train_data.shape[0]
            update_sign = train_y[i]*np.matmul(w, np.transpose(train_x[i]))
            # print(update_sign)
            if update_sign <= 0:
                w_array = np.append(w_array, w)
                c_array = np.append(c_array,c)
                w = w + r * train_y[i] * train_x[i]

                c = 1
            else:
                c += 1
    # print('w array :', w_array)
    # print('c array :', c_array)

    w_array = np.reshape(w_array,(c_array.shape[0], w.shape[1]))
    print('w array :', w_array)
    print('c array :', c_array)
    # print('w new is ', w)
    return w_array, c_array


def error(test_data, w_array, c_array):
    error = 0.0
    test_y = test_data[:, -1]
    test_x = np.copy(test_data)
    test_x[:, -1] = 1
    test_y[test_y == 0] = -1
    vote = np.zeros(test_data.shape[0], dtype=float)
    for i in range(0, c_array.shape[0]):
        w_array_i = np.reshape(w_array[i], (-1, test_data.shape[1]))
        # print(w_array_i)
        # print(w_array_i.shape[0])
        # print(w_array_i.shape[1])
        w_x = np.matmul(w_array_i, np.transpose(test_x))
        # print('w_x: ', w_x)
        # print(w_x.shape[0])
        # print(w_x.shape[1])
        w_x[w_x <= 0] = -1
        w_x[w_x > 0] = 1
        # print('w_x: ', w_x)
        c_w_x = c_array[i] * w_x
        # print('c_x_w: ',c_w_x)
        # print(c_w_x.shape[0])
        # print(c_w_x.shape[1])
        vote = vote + c_w_x
    # print('vote:', vote)
    # print(vote.shape[0])
    # print(vote.shape[1])
    # print(c_array.shape[0])
    vote[vote <= 0] = -1
    vote[vote > 0] = 1
    # print(vote)
    # print(vote.shape[0])
    # print(vote.shape[1])
    test_y = np.reshape(test_y, (-1, 1))
    vote = np.reshape(vote, (-1, 1))
    # print(vote)
    # print(vote.shape[0])
    # print(vote.shape[1])
    # print(vote[0])
    # print(test_y)
    for i in range(0, vote.shape[0]):
        if vote[i] != test_y[i]:
            error += 1

    error_rate = error / test_x.shape[0]
    print('error rate:', error_rate)
    return error_rate

train_data = np.loadtxt('train.csv', dtype='float32', delimiter=",")
w_array, c_array = voted_per(train_data)
# print(w_array.shape[0])
# print(w_array.shape[1])
# print(c_array.shape[0])
# print(w_array[1])
# print(w_array[1].shape[0])
# print(w_array[1].shape[1])

test_data = np.loadtxt('test.csv', dtype='float32', delimiter=",")
error_rate = error(test_data, w_array, c_array)


