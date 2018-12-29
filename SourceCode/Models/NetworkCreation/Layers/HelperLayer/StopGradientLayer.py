import tensorflow as tf

class StopGradientLayer():
    def __init__(self, supervisor, previous_size, size):
        self.size = size
        self.previous_size = previous_size
        self.supervisor = supervisor

    def __call__(self, list_of_inputs):
        outs = []
        for inp in list_of_inputs:
            outs.append(tf.stop_gradient(inp))
        return outs

    def default_output(self):
        default = []
        for s in self.size:
            defx = tf.placeholder_with_default(tf.fill([self.supervisor.batch_shape, s], 0.0),
                                               shape=[None, s])
            default.append(self.supervisor.store_op(defx, 'default_output'))
        return default