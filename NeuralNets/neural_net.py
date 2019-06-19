import numpy as np
import matplotlib.pyplot as plt

class MLPNet(object):

  """
  In this class we implement a MLP neural network. 
  H: hidden layer size
  N: input size
  D: Number of features
  C: class
  Loss Function: Softmax
  Regularization: L2 norm
  Activation Function: ReLU
  
  """
  def __init__(self, D, H, output_size, std=1e-4):
    """
    In this part we initialize the model as below:
    weights are initialize with small random value and biases are initialized with zero value. 
    these values are stored in the self.p_net as dictionary
    """
    self.p_net = {}
    self.p_net['W1'] = std * np.random.randn(D, H)
    self.p_net['b1'] = np.zeros(H)
    self.p_net['W2'] = std * np.random.randn(H, output_size)
    self.p_net['b2'] = np.zeros(output_size)
    ############################################
    self.H = H
    self.C = output_size
    self.gradient = {'W1': np.zeros(self.p_net['W1'].shape), 'W2': np.zeros(self.p_net['W2'].shape), 
                 'b1': np.zeros(self.p_net['b1'].shape), 'b2': np.zeros(self.p_net['b2'].shape)}
    
  def relu(self, x):
    if x > 0:
        return x
    return 0

  def softmax(self, X, indx):
    exp = np.exp(X[indx])
    return exp / np.sum(np.exp(X))

  def output_coding(self, y):
    output = np.zeros(self.C)
    output[int(y)] = 1
    return output

  def d_L_d_s(self, scores, Y, key):
    n = len(Y)
    for y, score, indx in zip(Y, scores, range(len(Y))):
        for i, s in enumerate(score[1]):
            if y == i:
                self.gradient[key][indx][i] = -1/n * (1 - self.softmax(score[1], i))
            else:
                self.gradient[key][indx][i] = 1/n * self.softmax(score[1], i)

  def d_s_d_w(self, XP, Weights, biases, sample_indx, key):
    for i, xp in enumerate(XP):
        for j, w in enumerate(np.transpose(Weights)):
            if np.sum(XP*w+biases[j]) <= 0:
                self.gradient[key][sample_indx][i][j] = 0
            else:
                self.gradient[key][sample_indx][i][j] = xp

  def d_s_d_b(self, X, biases, sample_indx, key):
    for i, x in enumerate(X):
        if x <= 0:
            self.gradient[key][sample_indx][i] = 0
        else:
            self.gradient[key][sample_indx][i] = 1

  def d_s_d_sp(self, XP, X, Weights, biases, sample_indx, key):
    for i, xp in enumerate(XP):
        for j, w in enumerate(np.transpose(Weights)):
            if np.sum(XP*w+biases[j]) <= 0:
                self.gradient[key][sample_indx][i][j] = 0
            else:
                self.gradient[key][sample_indx][i][j] = xp

  def loss(self, X, y=None, reg=0.0):

    """
      calculate the loss and its gradients for network:
      our inputs are:
        X: N*D matrix 
        y: training labels

      Returns:
      if y is empty :
        -return score matrix with shape (N,C) .each element of this matrix shows score for class c on input X[i]
      otherwise:
        -return a tuple of loss and gradient.
    """
    Weight2, bias2 = self.p_net['W2'], self.p_net['b2']
    Weight1, bias1 = self.p_net['W1'], self.p_net['b1']
    N, D = X.shape

    # forward pass
    scores = None
    #############################################################################
    # calculate output of each neurons
    # store results in the scores variable.
    scores = [[[0 for i in range(self.H)], [0 for i in range(self.C)]] for sample in X]
    for indx, sample in enumerate(X):
        for i in range(self.H):
            scores[indx][0] = np.matmul(X[indx], Weight1) + bias1
        for i in range(self.C):
            scores[indx][1] = np.matmul(scores[indx][0], Weight2) + bias2
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    if y is None:
      return scores

    # fill loss function.
    loss = None
    ############################################################################# 
    # loss = data loss + L2 regularization  
    loss = np.zeros(len(X))
    for x, label, i in zip(X, y, range(len(X))):
        loss[i] = -1 * np.log(self.softmax(scores[i][1], label)) + 1/2 * (np.sum(Weight1**2) + np.sum(Weight2**2))
    loss = np.sum(loss)
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # calculate gradients
    #############################################################################
    # store derivation of network's parameters(W and b) in the gradient
    # as dictionary structure    
    self.gradient['s2_w2'] = np.zeros((len(X), *Weight2.shape))
    self.gradient['s1_w1'] = np.zeros((len(X), *Weight1.shape))
    self.gradient['s2_s1'] = np.zeros((len(X), *Weight2.shape))
    self.gradient['s2_b2'] = np.zeros((len(X), Weight2.shape[1]))
    self.gradient['s1_b1'] = np.zeros((len(X), Weight1.shape[1]))
    self.gradient['L_s1'] = np.zeros((len(X), len(scores[0][0])))
    self.gradient['L_s2'] = np.zeros((len(X), len(scores[0][1])))
        
    self.d_L_d_s(scores, y, 'L_s2')  
    for i, x in enumerate(X):
        print(i)
        self.d_s_d_w(scores[i][0], Weight2, bias2, i, 's2_w2')
        self.d_s_d_w(x, Weight1, bias1, i,'s1_w1')
        self.d_s_d_sp(scores[i][0], scores[i][1], Weight2, bias2, i, 's2_s1')
        self.d_s_d_b(scores[i][1], bias2, i, 's2_b2')
        self.d_s_d_b(scores[i][0], bias1, i, 's1_b1')
        self.gradient['W2'] += self.gradient['s2_w2'][i] * self.gradient['L_s2'][i]
    self.gradient['W2'] += Weight2
    for i, x in enumerate(X):
        self.gradient['L_s1'][i] = np.matmul(self.gradient['L_s2'][i], np.transpose(self.gradient['s2_s1'][i]))
        self.gradient['W1'] += self.gradient['L_s1'][i] * self.gradient['s1_w1'][i]
        self.gradient['b2'] += self.gradient['L_s2'][i] * self.gradient['s2_b2'][i]
        self.gradient['b1'] += self.gradient['L_s1'][i] * self.gradient['s1_b1'][i]
    self.gradient['W1'] += Weight1
    
    
            
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, self.gradient

  def train(self, X, y, X_val, y_val,
            alpha=1e-3, alpha_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=100):

    """
    We want to train this network with stochastic gradient descent.
    Our inputs are:

    - X: array of shape (N,D) for training data.
    - y: training labels.
    - X_val: validation data.
    - y_val: validation labels.
    - alpha: learning rate
    - alpha_decay: This factor used to decay the learning rate after each epoch
    - reg: That shows regularization .
    - num_iters: Number of epoch 
    - batch_size: Size of each batch

    """
    num_train = X.shape[0]
    iteration = max(num_train / batch_size, 1)

    loss_train = []
    train_acc = []
    val_acc = []

    for it in range(num_iters):
        print('iteration '+str(it))
        data_batch = None
        label_batch = None
        
        #########################################################################
        # create a random batch of data and labels for
        indx = np.random.permutation(len(X))
        data, labels = X[indx], y[indx]
        data_batch = data[0:batch_size]
        label_batch = labels[0:batch_size]
        #########################################################################
        pass
        #########################################################################
        #                             END OF YOUR CODE                          #
        #########################################################################
        # calculate loss and gradients
        loss, gradient = self.loss(data_batch, y=label_batch, reg=reg)
        loss_train.append(loss)
        #########################################################################
        # update weights and biases which stored in the slef.p_net regarding 
        # to gradient dictionary.
        self.p_net['W1'] -= alpha * gradient['W1']
        self.p_net['b1'] -= alpha * gradient['b1']
        self.p_net['W2'] -= alpha * gradient['W2']
        self.p_net['b2'] -= alpha * gradient['b2']
        #########################################################################
        pass
        #########################################################################
        #                             END OF YOUR CODE                          #
        #########################################################################
        if it % 100 == 0:
            print ('iteration %d / %d: loss %f' % (it, num_iters, loss))
        
        if it % iteration == 0:
        # Check accuracy
            train_acc_ = (self.predict(data_batch) == label_batch).mean()
            val_acc_ = (self.predict(X_val) == y_val).mean()
            train_acc.append(train_acc_)
            val_acc.append(val_acc_)

            alpha *= alpha_decay

    return {
      'loss_train': loss_train,
      'train_acc': train_acc,
      'val_acc': val_acc,
    }

  def predict(self, X):

    """
    After you train your network use its parameters to predict labels

    Returns:
    - y_prediction: array which shows predicted lables
    """
    y_prediction = []

    ###########################################################################
    # Implement this function. thats VERY easy to do
    for i, x in enumerate(X):
        l1 = np.matmul(x, self.p_net['W1']) + self.p_net['b1']
        l1 = np.array([self.relu(s) for s in l1])
        l2 = np.matmul(l1, self.p_net['W2']) + self.p_net['b2']
        l2 = np.array([self.relu(s) for s in l2])
        y_prediction.append(np.argmax(l2))
    ###########################################################################
    pass
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_prediction


