import tensorflow as tf

# Takes one input which is the previous layer
class SampleNormalLayer():
    def __init__(self, supervisor, previous_size, size):
        self.prev_inp_size = previous_size[0]
        self.size = size
        self.supervisor = supervisor

    def __call__(self, list_of_inputs):
        means = list_of_inputs[0]
        vars = list_of_inputs[1]
        sampled_act = (means +
                       tf.exp(vars / 2.0) *
                       tf.random_normal(shape=(self.prev_inp_size,)))
        return [sampled_act]

    def default_output(self):
        def1 = tf.placeholder_with_default(tf.fill([self.supervisor.batch_shape, self.size], 0.0),
                                           shape=[None, self.size])
        return [self.supervisor.store_op(def1, 'default_output')]