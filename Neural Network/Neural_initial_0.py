import numpy as np

def sigmoid_function(s):
    sigmoid = 1 / (1 + np.exp(-s))
    return sigmoid


def sigmoid_derivation(s):
    sigmoid = 1 / (1 + np.exp(-s))
    derivation = sigmoid * (1 - sigmoid)
    return derivation

def loss_function(y, out):
    J1 = np.matmul(y, np.log(out).T)
    J2 = np.matmul((1-y), np.log(1-out).T)
    J = - J1-J2
    return J

def LearningRate_new(r0, t):
    r = r0 / (1 + (r0/0.1)*t)
    return r

def SGD_zero(x, y, L_size):
    LearnRate = 0.01
    train_x = np.copy(x)
    w_set = {}
    num_y = y.shape[0]
    for i in range(len(L_size)):

        if i == 0:
            W = np.zeros((L_size[i], x.shape[0]))
            b = np.zeros((L_size[i], 1))
        else:
            W = np.zeros((L_size[i], L_size[i - 1]))
            b = np.zeros((L_size[i], 1))
        w_set['W' + str(i + 1)] = W
        w_set['b' + str(i + 1)] = b
    W = np.random.normal(0, 1, (num_y, L_size[-1]))
    b = np.zeros((num_y, 1))
    w_set['W' + str(len(L_size) + 1)] = W
    w_set['b' + str(len(L_size) + 1)] = b
    J_set = []
    num_x = np.arange(train_x.shape[1])
    for t in range(0,50):
        np.random.shuffle(num_x)
        x = x[:, num_x]
        y = y[:, num_x]
        for i in range(train_x.shape[1]):
            output, layers = forward_pass(x[:, i], w_set)
            J = loss_function(y[:, i], output)
            grad_weights = backpropagation(y[:, i], output, w_set, layers)
            LearnRate = LearningRate_new(LearnRate, t)
            for j in range(0, 3):
                w_set['W' + str(j + 1)] = w_set['W' + str(j + 1)] - LearnRate * grad_weights['dW' + str(j + 1)]
                w_set['W' + str(j + 1)] = w_set['W' + str(j + 1)]
                w_set['b' + str(j + 1)] = w_set['b' + str(j + 1)] - LearnRate * grad_weights['db' + str(j + 1)]
                w_set['b' + str(j + 1)] = w_set['b' + str(j + 1)]

            J_set = np.append(J_set,J)
    # print(w_set)

    return w_set

def forward_pass(x, w_set):


    z_b = np.reshape(x,(-1,1))
    layers = dict()
    layers['S0'], layers["Z_b0"] = x[:, np.newaxis], x[:, np.newaxis]
    i = 0

    for key, value in w_set.items():
        if key == 'W'+str(i+1):
            s = np.matmul(value,z_b)
            z_b = sigmoid_function(s)
            layers['S' + str(i+1)] = s
            layers['Z_b' + str(i+1)] = z_b
            i +=1


    return z_b, layers




def backpropagation(y, output, w_set, layers):
    gradient = {}
    dL_dz = output - y
    for i in reversed(range(0, 3)):
        dy_dw = layers['Z_b' + str(i)]
        dL_dw = np.matmul(dL_dz, dy_dw.T)
        db = np.sum(dL_dz, axis=1)
        gradient['dW' + str(i + 1)] = dL_dw
        gradient['db' + str(i + 1)] = db
        dL_dz_new= np.dot(w_set['W' + str(i + 1)].T, dL_dz)
        z_layer = layers['S' + str(i)]
        dL_dz = dL_dz_new * sigmoid_derivation(z_layer)


    return gradient



def error_rate(x, y, w_set):
    z_b = np.copy(x)
    for i in range(len(w_set) // 2):
        W = w_set['W' + str(i + 1)]
        s = np.dot(W, z_b)
        z_b = sigmoid_function(s)

    predict = np.where(z_b >= 0.5, 1, 0)
    m = y.shape[1]
    error = 1 - np.sum(y == predict) / m
    return error


train_data = np.loadtxt('train.csv', dtype='float32', delimiter=",")
test_data = np.loadtxt('test.csv', dtype='float32', delimiter=",")

widths = [5, 10, 25, 50, 100]

train_x = train_data[:, :-1]
train_y = train_data[:, -1]
train_y = np.reshape(train_y, (-1,1))Compare
train_x, train_y = train_x.T, train_y.T

test_x = test_data[:, :-1]
test_y = test_data[:, -1]
test_y = np.reshape(test_y, (-1,1))
test_x, test_y = test_x.T, test_y.T

m = 1
print('when the initial weights are 0')
for w in widths:
    w_set = SGD_zero(train_x, train_y, L_size=[w, w])
    train_error = error_rate(train_x, train_y, w_set)
    test_error = error_rate(test_x, test_y, w_set)
    print(m, '. if the width is ',w)
    print('Train Error is: ', train_error)
    print('Test Error is: ', test_error)
    m += 1