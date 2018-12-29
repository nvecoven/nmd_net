import tensorflow as tf

# Takes two input which is the previous layer
class GruLayer():
    def __init__(self, supervisor, previous_size, size):
        self.prev_inp_size = previous_size[0]
        self.prev_state_size = previous_size[1]
        self.size = size
        self.supervisor = supervisor
        self.iwz = supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[self.prev_inp_size, self.size]), name = 'iwz')
        self.iwr = supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[self.prev_inp_size, self.size]), name = 'iwr')
        self.iwh = supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[self.prev_inp_size, self.size]), name = 'iwh')
        self.swz = supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[self.prev_state_size, self.size]), name = 'swz')
        self.swr = supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[self.prev_state_size, self.size]), name = 'swr')
        self.swh = supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[self.prev_state_size, self.size]), name = 'swh')
        self.bz = supervisor.variable(tf.constant(value=0.0, shape =[size]), name = 'bz')
        self.br = supervisor.variable(tf.constant(value=0.0, shape = [size]), name = 'br')

    def __call__(self, list_of_inputs):
        inp = list_of_inputs[0]
        prev_out = list_of_inputs[1]
        z = tf.nn.sigmoid(tf.matmul(inp, self.iwz) + tf.matmul(prev_out, self.swz) + self.bz)
        r = tf.nn.sigmoid(tf.matmul(inp, self.iwr) + tf.matmul(prev_out, self.swr) + self.br)
        h = tf.nn.tanh(tf.matmul(inp, self.iwh) + tf.multiply(r, tf.matmul(prev_out, self.swh)))
        output = (1.0-z) * h + z * prev_out
        return [output]

    def default_output(self):
        def1 = tf.placeholder_with_default(tf.fill([self.supervisor.batch_shape, self.size], 0.0),
                                           shape=[None, self.size])
        return [self.supervisor.store_op(def1, 'default_output')]