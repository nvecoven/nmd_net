import tensorflow as tf

class InputLayer():
    def __init__(self, supervisor, previous_size, size):
        self.size = size
        self.supervisor = supervisor

    def __call__(self, list_of_inputs):
        return list_of_inputs

    def default_output(self):
        def1 = tf.placeholder_with_default(tf.fill([self.supervisor.batch_shape, self.size], 0.0),
                                           shape=[None, self.size])
        return [self.supervisor.store_op(def1, 'default_output')]