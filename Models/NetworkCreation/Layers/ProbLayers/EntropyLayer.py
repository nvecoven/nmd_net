import tensorflow as tf
import numpy as np

# Takes one input which is the previous layer
class EntropyLayer():
    def __init__(self, supervisor, previous_size, size, params):
        self.prev_inp_size = previous_size[0]
        self.size = size
        self.supervisor = supervisor
        self.act_dim = params[0]

    def __call__(self, list_of_inputs):
        log_vars = list_of_inputs[0]
        # entropy = tf.reduce_mean(0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
        #                                 tf.reduce_sum(log_vars, axis=1)))
        entropy = tf.reshape(0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
                         tf.reduce_sum(log_vars, axis=1)),[-1,1])
        return [entropy]

    def default_output(self):
        def1 = tf.placeholder_with_default(tf.fill([self.supervisor.batch_shape, self.size], 0.0),
                                           shape=[None, self.size])
        return [self.supervisor.store_op(def1, 'default_output')]