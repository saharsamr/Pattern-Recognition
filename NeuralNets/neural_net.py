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
    
  def relu(self, x):
    if x > 0:
        return x
    return 0

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
            print(Weight1.shape)
            scores[indx][0] = np.matmul(X[indx], Weight1)
        for i in range(self.C):
            print(Weight2)
            scores[indx][1] = np.matmul(scores[indx][0], Weight2)
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
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # calculate gradients
    gradient = {}
    #############################################################################
    # store derivation of network's parameters(W and b) in the gradient
    # as dictionary structure
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, gradient

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
      data_batch = None
      label_batch = None

      #########################################################################
      # create a random batch of data and labels for training store 
      # them into data_batch and label_batch  
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
      #########################################################################
      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if it % 100 == 0:
        print ('iteration %d / %d: loss %f' % (it, num_iters, loss))

      if it % iteration == 0:
        # Check accuracy
        train_acc = (self.predict(data_batch) == label_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc.append(train_acc)
        val_acc.append(val_acc)

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
    y_prediction = None

    ###########################################################################
    # Implement this function. thats VERY easy to do
    ###########################################################################
    pass
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_prediction


