from nets.base_net import BaseNet
from data import dataloader
from activate_fn import activate as act_fn
import tensorflow as tf
import tensorlayer as tl
from models.model import Model

class Net(BaseNet):
  '''
  '''
  def __init__(self, in_shape = [224, 224, 3]):
    '''
    '''
    super().__init__(in_shape = in_shape)
    return

  def _build_network(self):
    '''
    '''
    network = tl.layers.InputLayer(self._inputs, name = 'input_layer')
    network = tl.layers.Conv2d(network, 32, (5, 5), (2, 2), act = tf.nn.relu, padding = 'VALID', name = 'cnn1')
    network = tl.layers.MaxPool2d(network, (2, 2), (2, 2), padding = 'VALID', name = 'pool1')
    network = tl.layers.Conv2d(network, 128, (5, 5), (3, 3), act = tf.nn.relu, padding = 'VALID', name = 'cnn2')
    network = tl.layers.MaxPool2d(network, (2, 2), (1, 1), padding = 'VALID', name = 'pool2')
    network = tl.layers.Conv2d(network, 256, (1, 1), (1, 1), act = tf.nn.relu, padding = 'VALID', name = 'cnn3')
    network = tl.layers.Conv2d(network, 512, (5, 5), (2, 2), act = tf.nn.relu, padding = 'VALID', name = 'cnn4')
    network = tl.layers.MaxPool2d(network, (2, 2), (2, 2), padding = 'VALID', name = 'pool4')
    network = tl.layers.FlattenLayer(network, name = 'flatten4')
    network = tl.layers.DenseLayer(network, n_units = 2, act = tf.nn.relu, name = 'outputs')
    return network

  def load_dataset(self, filename = None):
    '''
    @brief: load dataset and return X_train, Y_train, x_val, y_val, x_test, y_test
    '''
    return None

def main(_):
  '''
  '''
  
  return

if __name__ == '__main__':
  tf.app.run()

