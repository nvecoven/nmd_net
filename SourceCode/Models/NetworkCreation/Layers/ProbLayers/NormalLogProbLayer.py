import tensorflow as tf

# Takes one input which is the previous layer
class NormalLogProbLayer():
    def __init__(self, supervisor, previous_size, size):
        self.prev_inp_size = previous_size[0]
        self.size = size
        self.supervisor = supervisor

    def __call__(self, list_of_inputs):
        means = list_of_inputs[0]
        log_vars = list_of_inputs[1]
        action = list_of_inputs[2]

        logp = -0.5 * tf.reduce_sum(log_vars, axis = 1)
        logp += -0.5 * tf.reduce_sum(tf.square(action - means)/(tf.exp(log_vars)+1e-6), axis = 1)
        logp = tf.reshape(logp, [-1, 1])
        return [logp]

    def default_output(self):
        def1 = tf.placeholder_with_default(tf.fill([self.supervisor.batch_shape, self.size], 0.0),
                                           shape=[None, self.size])
        return [self.supervisor.store_op(def1, 'default_output')]