import tensorflow as tf

# Takes one input which is the previous layer
class AdaptiveLogitsLayer2():
    def __init__(self, supervisor, previous_size, size):
        self.prev_inp_size = previous_size[0]
        self.nmd_feat_size = previous_size[1]
        self.size = size
        self.supervisor = supervisor
        self.w = supervisor.variable(tf.truncated_normal(stddev=0.1, shape =[self.prev_inp_size, size]), name = 'w')
        self.b = supervisor.variable(tf.constant(value=0.0, shape = [size]), name = 'b')
        self.nmd_matrix_multbias = supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[self.nmd_feat_size, self.size]))
        self.nmd_matrix_transbias = supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[self.nmd_feat_size, self.size]))
        # self.ab = supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[size]), name = 'ab')
        # self.mb = supervisor.variable(tf.constant(value=1.0, shape=[size]), name = 'mb')
        # self.amb = supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[size]), name = 'amb')

    def __call__(self, list_of_inputs):
        inp = list_of_inputs[0]
        nmd_features = list_of_inputs[1]

        weights = self.w
        mult_bias = tf.matmul(nmd_features, self.nmd_matrix_multbias) + 1.0
        bias = self.b
        output = tf.matmul(inp, weights) * mult_bias + bias + tf.matmul(nmd_features, self.nmd_matrix_transbias)
        return [output]

    def default_output(self):
        def1 = tf.placeholder_with_default(tf.fill([self.supervisor.batch_shape, self.size], 0.0),
                                           shape=[None, self.size])
        return [self.supervisor.store_op(def1, 'default_output')]