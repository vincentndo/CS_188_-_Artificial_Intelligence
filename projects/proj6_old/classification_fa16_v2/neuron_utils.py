import numpy as np
from util import raiseNotDefined

def gradient(f, w):
    """YOUR CODE HERE"""

    ret = np.zeros(w.shape, "float")
    for i in range(len(w)):
      delta = max(w[i] / 1000.0, 0.001)
      dummy_w1, dummy_w2 = w.copy(), w.copy()
      dummy_w1[i] = dummy_w1[i] - delta
      dummy_w2[i] = dummy_w2[i] + delta
      ret[i] = (f(dummy_w2) - f(dummy_w1)) / (2 * delta)
    return ret


def sanity_check_gradient():
  """Handy function for debugging your gradient method."""
  def g(w):
    w1 = w[0]
    w2 = w[1]
    return w1 ** 3 * w2 + 3 * w1
  
  print("The print statement below should output approximately [111, 27]")
  print(gradient(g, np.array([3, 4], dtype='f')))

  def loss(self, sample, label, w):
      """sample: np.array of shape(1, numFeatures).
      label:  the correct label of the sample 
      w:      the weight vector under which to calculate loss

      Can interpret loss as the probability of sample being in the correct class when
      classified by a SigmoidNeuron. 

      For numerical accuracy reasons, the loss is expressed as 
      math.log(1/sigmoid) instead of -math.log(sigmoid) as we discussed in class.

      Do not modify this function."""
      z = np.dot(w, sample)

      if label == True:
          return math.log(1.0 + math.exp(-2*z))
      else:
          return math.log(1.0 + math.exp(2*z))

sanity_check_gradient()