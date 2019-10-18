import tensorflow as tf

# Takes one input which is the previous layer
class AdaptiveAPLLayer():
    def __init__(self, supervisor, previous_size, size, params):
        nbr_hinges = params[0]
        if len(params) > 1:
            regul_var_nbr = params[1]
        else:
            regul_var_nbr = supervisor.net_number
        self.prev_inp_size = previous_size[0]
        self.adapt_features_size = previous_size[1]
        self.size = size
        self.supervisor = supervisor
        self.w = supervisor.variable(tf.truncated_normal(stddev=0.1, shape =[self.prev_inp_size, size]), name = 'w')
        self.b = supervisor.variable(tf.constant(value=0.0, shape = [size]), name = 'b')
        self.has = []
        self.a_has = []
        self.hbs = []
        self.a_hbs = []
        self.was = []
        self.wbs = []
        for i in range(nbr_hinges):
            self.has.append(supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[size]), var_list = regul_var_nbr,
                                                name = 'has'))
            self.hbs.append(supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[size]), var_list = regul_var_nbr,
                                                name = 'hbs'))
            self.a_has.append(supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[size]), var_list = regul_var_nbr,
                                                  name = 'a_has'))
            self.a_hbs.append(supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[size]), var_list = regul_var_nbr,
                                                  name = 'a_hbs'))
            self.was.append(
                supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[self.adapt_features_size, size]),
                                    name = 'was'))
            self.wbs.append(
                supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[self.adapt_features_size, size]),
                                    name = 'wbs'))

    def __call__(self, list_of_inputs):
        inp = list_of_inputs[0]
        adapt_features = list_of_inputs[1]
        x = tf.matmul(inp, self.w) + self.b
        linear_pieces = [tf.nn.relu(x)]
        for a,b,aa,ab,wa,wb in zip(self.has, self.hbs, self.a_has, self.a_hbs, self.was, self.wbs):
            # Adapation here
            cur_a_adapt = tf.nn.tanh(tf.matmul(adapt_features, wa))
            cur_b_adapt = tf.nn.tanh(tf.matmul(adapt_features, wb))
            ra = a + cur_a_adapt * aa
            rb = b + cur_b_adapt * ab
            piece = tf.multiply(ra, tf.nn.relu(tf.add(-x, rb)))
            linear_pieces.append(piece)
        output = tf.add_n(linear_pieces)
        return [output]

    def default_output(self):
        def1 = tf.placeholder_with_default(tf.fill([self.supervisor.batch_shape, self.size], 0.0),
                                           shape=[None, self.size])
        return [self.supervisor.store_op(def1, 'default_output')]