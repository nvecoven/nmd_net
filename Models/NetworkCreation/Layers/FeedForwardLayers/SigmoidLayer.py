import tensorflow as tf

# Takes one input which is the previous layer
class SigmoidLayer():
    def __init__(self, supervisor, previous_size, size):
        self.prev_inp_size = previous_size[0]
        self.size = size
        self.supervisor = supervisor
        self.w = supervisor.variable(tf.truncated_normal(stddev=0.1, shape =[self.prev_inp_size, size]), name = 'w')
        self.b = supervisor.variable(tf.constant(value=0.0, shape = [size]), name = 'b')

    def __call__(self, list_of_inputs):
        inp = list_of_inputs[0]
        x = tf.add(tf.matmul(inp, self.w), self.b)
        # out = tf.clip_by_value(x, clip_value_min=-1.0, clip_value_max=1.0)
        out = tf.nn.sigmoid(x)
        return [out]

    def default_output(self):
        def1 = tf.placeholder_with_default(tf.fill([self.supervisor.batch_shape, self.size], 0.0),
                                           shape=[None, self.size])
        return [self.supervisor.store_op(def1, 'default_output')]