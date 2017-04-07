import numpy as np
from random import shuffle
from math import log, exp

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_trains = X.shape[0]
  num_classes = W.shape[1]
  for n in range(num_trains):
        s = np.dot(X[n], W)
        # avoid numerical stability problem by subtracting the max
        s -= np.max(s)
        expS = np.exp(s)
        p = expS/np.sum(expS)
        loss += -log( p[y[n]] )
        for j in range(num_classes):
            dW[:,j] += p[j]*X[n]
            if j == y[n]:
                dW[:,j] += - X[n]
            
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_trains
  loss += 0.5*reg*np.sum(W*W)

  dW /= num_trains
  dW += reg*W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_trains = X.shape[0]
  num_classes = W.shape[1]
  
  S = np.dot(X,W)
  S -= S.max(axis = 1, keepdims = True)
  expS = np.exp(S)

  P = expS/expS.sum(axis = 1, keepdims = True )
  L = P[range(num_trains), y]
  loss = -np.log(L).sum()

  P[range(num_trains), y]  -= 1.0
  dW = np.dot(X.T, P)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_trains
  loss += 0.5*reg*np.sum(W*W)

  dW /= num_trains
  dW += reg*W
  return loss, dW

