import numpy as np


class BaseModel(object):
  '''Base Model for kaggle-mnist, your model should inherit this model
  and implement `train` and `test` method
  '''

  def __init__(self, name='base'):
    self.name = 'base'

  def train(self, X, Y):
    '''train the model with given data (X, Y)
    Parameters
    ==========
    X: data, shape: (N, 28, 28), dtype=np.float32
    Y: label, shape: (N,), dtype=np.int32

    Returns
    =======
    None
    '''
    raise NotImplemented('Subclass should implement train')

  def predict(self, X):
    '''predict with given data (X)
    Parameters
    ==========
    X: data, shape: (N, 28, 28), dtype=np.float32

    Returns
    =======
    Y: your model's prediction, shape: (N,), dtype=np.int32
    '''
    raise NotImplemented('Subclass should implement predict')


class DummyModel(BaseModel):
  '''this is a dummy model
  during train, it calculate a mean image for every label, during test, every data sample compare
  with these mean image using L2 distance to detect which label it is.
  '''

  def __init__(self):
    super(DummyModel, self).__init__('dummy')

  def train(self, X, Y):
    self.ws = [np.zeros((28, 28), dtype=np.float32) for i in range(10)]
    self.ns = [0 for i in range(10)]
    for idx, (x, y) in enumerate(zip(X, Y)):
      assert y >= 0 and y < 10
      self.ws[y] += x
      self.ns[y] += 1
    for i in range(10):
      if self.ns[i] > 0:
        self.ws[i] /= self.ns[i]

  def predict(self, X):
    Y = np.zeros(len(X), dtype=np.int32)
    dist_func = lambda x1, x2: np.sum(np.square(x1-x2))
    for i, x in enumerate(X):
      dist_min, label = np.finfo(np.float32).max, -1
      for j, w in enumerate(self.ws):
        dist = dist_func(x, w)
        if dist < dist_min:
          dist_min = dist
          label = j
      assert label != -1
      Y[i] = label
    return Y


###############################################
#             Put your model below            #
###############################################
