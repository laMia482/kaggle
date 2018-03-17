import os
import csv
import numpy as np
from conf.conf import config as cfg
from utils import dataloader
from nets.base_net import BaseNet
from models.model import Model
from activate_fn import activate as act_fn
import tensorflow as tf
import tensorlayer as tl

class Net(BaseNet):
  '''
  '''
  def __init__(self, in_shape = [5]):
    '''
    '''
    super().__init__(in_shape = in_shape)
    return

  def _build_network(self):
    '''
    '''
    network = tl.layers.InputLayer(self._inputs, name = 'input_layer')
    network = tl.layers.FlattenLayer(network, name = 'flatten1')
    end_points = [network]
    for id, val in enumerate([256, 1024, 512, 256, 1024, 512, 512, 1024]):
      # network = tl.layers.DropoutLayer(network, keep = 0.8, is_fix = True, name = 'drop' + str(id))
      network = tl.layers.DenseLayer(network, n_units = val, act = tf.nn.relu, name = 'dense' + str(id))
      end_points += [network]
    network = tl.layers.ConcatLayer(end_points, 1, name = 'concat')
    network = tl.layers.DenseLayer(network, n_units = 512, act = act_fn.l2norm, name = 'dense')
    # network = tl.layers.DropoutLayer(network, keep = 0.8, is_fix = True, name = 'drop')
    network = tl.layers.DenseLayer(network, n_units = 2, act = tf.identity, name = 'output')
    return network

  def load_dataset(self, filename = None):
    '''
    @brief: load dataset and return X_train, Y_train, x_val, y_val, x_test, y_test
    '''
    if filename is None:
      raise Exception('file is not valid in load_dataset')
    X, Y = [], [] # pClass, sex, age, sibsp, parch -> survived
    with open(filename, encoding = 'utf-8') as fp:
      csv_reader = csv.reader(fp)
      for id, row in enumerate(csv_reader):
        if id == 0:
          continue
        col = []
        frac = [1., 1. / 5., 1., 1.]
        for col_id, col_val in enumerate([2, 5, 6, 7]):
          val = 0 if row[col_val] == '' else frac[col_id] * float(row[col_val])
          col += [val]
        col += [0.0 if row[4] == 'male' else 1.0]
        x = col
        # for col_1 in col:
        #   for col_2 in col:
        #     x.append(col_1 * col_2)
        X.append(x)
        y = 0 if row[1] == '' else int(row[1])
        Y.append(y)
    # for id, val in enumerate(x):
      # print('id: {}, val: {}'.format(id, val) + ', {}'.format(y[id]))
    return np.asarray(X), np.asarray(Y)

def main(_):
  '''
  '''
  model = Model(cfg = cfg, input_network = Net)
  x, y = model.get_model().load_dataset(filename = os.path.join('data', 'train.csv'))
  x1, y1 = x[:450], y[:450]
  x2, y2 = x[450:], y[450:]
  # np.random.shuffle(x)
  # np.random.shuffle(y)
  # train
  if cfg.is_train is True:
    for lr in [1.0, 0.1, 0.01, 1.0, 0.001, 0.01]:
      print('learning rate turns to {}'.format(lr))
      model.train(x = x1, y = y1, learning_rate = lr, x_test = x2, y_test = y2, batch_size_test = y2.shape[0])
  # eval
  if cfg.is_eval is True:
    model.eval(x2, y2)
  # test
  if cfg.is_predict is True:
    model.load_npz(os.path.join('ckpts', 'model.npz'))
    x_test, _ = model.get_model().load_dataset(filename = os.path.join('data', 'test.csv'))
    y_test = model.predict(x_test)
    # write into csv file
    with open('res.csv', 'w') as fp:
      writer = csv.writer(fp)
      writer.writerow(['PassengerId', 'Survived'])
      for id, val in enumerate(y_test):
        writer.writerow([892 + id, val])
  return

if __name__ == '__main__':
  tf.app.run()

