import os
import numpy as np
import pandas as pd


def load_train_data(path):
  '''load train data
  '''
  if not os.path.exists(path):
    raise RuntimeError("%s doesn't exists"%path)
  df = pd.read_csv(path)
  data = df.values
  label = data[:, 0]
  image = data[:, 1:]
  image = image.reshape((len(image), 28, 28))
  image = image.astype(np.float32)
  label = label.astype(np.int32)
  return image, label


def load_test_data(path):
  '''load test data
  '''
  if not os.path.exists(path):
    raise RuntimeError("%s doesn't exists"%path)
  df = pd.read_csv(path)
  image = df.values
  image = image.reshape((len(image), 28, 28))
  image = image.astype(np.float32)
  return image


def evaluate(Y, Y_gt):
  '''evaluate the result
  Parameters
  ==========
  Y: your prediction
  Y_gt: the ground truth

  Returns
  =======
  acc for every label
  '''
  acc = np.zeros(10)
  num = np.zeros(10)
  for y, y_gt in zip(Y, Y_gt):
    num[y_gt] += 1
    if y == y_gt:
      acc[y_gt] += 1
  acc /= num
  return acc


def save_answer(Y, path='data/submission.csv'):
  idx = np.arange(len(Y)) + 1
  data = {'ImageId': idx,
          'Label': Y}
  df = pd.DataFrame(data=data)
  with open(path, 'w') as fout:
    fout.write(df.to_csv(index=False))
