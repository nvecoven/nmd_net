import tensorflow as tf

# Takes one input which is the previous layer
class KlLayer():
    def __init__(self, supervisor, previous_size, size, params):
        self.prev_inp_size = previous_size[0]
        self.size = size
        self.supervisor = supervisor
        self.act_dim = params[0]

    def __call__(self, list_of_inputs):
        old_log_vars = list_of_inputs[0]
        log_vars = list_of_inputs[1]
        old_means = list_of_inputs[2]
        means = list_of_inputs[3]

        log_det_cov_old = tf.reduce_sum(old_log_vars, axis = 1)
        log_det_cov_new = tf.reduce_sum(log_vars, axis = 1)
        tr_old_new = tf.reduce_sum(tf.exp(old_log_vars - log_vars), axis = 1)
        # kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                  # tf.reduce_sum(tf.square(means - old_means) /
                                  #               tf.exp(log_vars), axis=1) - self.act_dim)
        kl = tf.reshape(0.5 * (log_det_cov_new - log_det_cov_old + tr_old_new +\
             tf.reduce_sum(tf.square(means - old_means) /
                           (tf.exp(log_vars)+1e-6), axis=1) - self.act_dim),[-1,1])
        # TODO : Check that kl is a scalar in output
        return [kl]

    def default_output(self):
        def1 = tf.placeholder_with_default(tf.fill([self.supervisor.batch_shape, self.size], 0.0),
                                           shape=[None, self.size])
        return [self.supervisor.store_op(def1, 'default_output')]