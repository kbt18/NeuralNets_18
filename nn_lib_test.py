import unittest
import numpy as np
from nn_lib import *

from src.nn_lib import LinearLayer

#hello
class TestLinearLayer(unittest.TestCase):

    def test_as_recommended_sizes(self):
        batch_size = 1
        n_in = 3
        n_out = 42
        learning_rate = 0.01
        layer = LinearLayer(n_in=n_in, n_out=n_out)
        inputs = np.random.rand(batch_size, n_in)
        outputs = layer(inputs)
        self.assertEqual(outputs.shape[0], batch_size)
        self.assertEqual(outputs.shape[1], n_out)

        grad_loss_wrt_outputs = np.random.rand(batch_size, n_out)
        grad_loss_wrt_inputs = layer.backward(grad_loss_wrt_outputs)
        self.assertEqual(grad_loss_wrt_inputs.shape[0], batch_size)
        self.assertEqual(grad_loss_wrt_inputs.shape[1], n_in)
        layer.update_params(learning_rate)
        pass

    def test_1x1(self):
        batch_size = 1
        n_in = 1
        n_out = 1
        learning_rate = 0.01
        layer = LinearLayer(n_in=n_in, n_out=n_out)
        layer._W = 2
        layer._b = 3
        inputs = [[1.0]]
        outputs = layer(inputs)
        self.assertEqual(outputs.shape[0], batch_size)
        self.assertEqual(outputs.shape[1], n_out)
        self.assertEqual(outputs, [[5.0]])  # 2*1 + 3

        grad_loss_wrt_outputs = np.random.rand(batch_size, n_out)

        grad_loss_wrt_outputs = np.zeros((batch_size, n_out))
        grad_loss_wrt_inputs = layer.backward(grad_loss_wrt_outputs)
        # self.assertEqual(grad_loss_wrt_inputs.shape[0], batch_size)
        # self.assertEqual(grad_loss_wrt_inputs.shape[1], n_in)
        # layer.update_params(learning_rate)
        # pass

    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper() or True)
    #
    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

if __name__ == '__main__':
    unittest.main()
