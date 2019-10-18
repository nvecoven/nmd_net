import tensorflow as tf

# Takes one input which is the previous layer
class AdaptiveLogitsLayer():
    def __init__(self, supervisor, previous_size, size):
        self.prev_inp_size = previous_size[0]
        self.size = size
        self.supervisor = supervisor
        self.w = supervisor.variable(tf.truncated_normal(stddev=0.1, shape =[self.prev_inp_size, size]), name = 'w')
        self.b = supervisor.variable(tf.constant(value=0.0, shape=[size]), name = 'b')
        self.ab = supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[size]), name = 'ab')
        self.mb = supervisor.variable(tf.constant(value=1.0, shape=[size]), name = 'mb')
        self.amb = supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[size]), name = 'amb')
    def __call__(self, list_of_inputs):
        inp = list_of_inputs[0]
        prev_sb = list_of_inputs[1]
        prev_smb = list_of_inputs[2]

        weights = self.w
        mult_bias = self.mb + self.amb * prev_smb
        bias = self.b + self.ab * prev_sb
        output = (tf.matmul(inp, weights) + bias) * mult_bias
        return [output]

    def default_output(self):
        def1 = tf.placeholder_with_default(tf.fill([self.supervisor.batch_shape, self.size], 0.0),
                                           shape=[None, self.size])
        return [self.supervisor.store_op(def1, 'default_output')]