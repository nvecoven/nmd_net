import tensorflow as tf

# Takes two input which is the previous layer
class HessianPlasticLayer():
    def __init__(self, supervisor, previous_size, size):
        self.prev_inp_size = previous_size[0]
        self.prev_state_size = previous_size[1]
        self.size = size
        self.supervisor = supervisor
        self.w = supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[self.prev_inp_size, self.size]), name = 'w')
        self.b = supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[self.prev_inp_size, self.size]), name = 'b')
        self.eta = supervisor.variable(tf.constant(value = 0.5, shape=[1]), name = 'eta')
        self.alpha = supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[self.prev_inp_size, self.size]), name = 'alpha')

    def __call__(self, list_of_inputs):
        #1 TODO FOR FUTURE TESTS
        inp = list_of_inputs[0]
        prev_out = list_of_inputs[1]
        prev_hess = list_of_inputs[2]

        tmp = tf.matmul(inp, self.w) + tf.matmul(inp, self.alpha * prev_hess)
        out = tf.clip_by_value(tmp, clip_value_min=-1.0, clip_value_max=1.0)

        new_hess = prev_hess + self.eta*prev_out*()
        return [out, ]

    def default_output(self):
        def1 = tf.placeholder_with_default(tf.fill([self.supervisor.batch_shape, self.size], 0.0),
                                           shape=[None, self.size])
        return [self.supervisor.store_op(def1, 'default_output')]