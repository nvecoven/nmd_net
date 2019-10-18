import tensorflow as tf

# Takes one input which is the previous layer
class APLLayer():
    def __init__(self, supervisor, previous_size, size, params):
        nbr_hinges = params[0]
        self.prev_inp_size = previous_size[0]
        self.size = size
        self.supervisor = supervisor
        self.w = supervisor.variable(tf.truncated_normal(stddev=0.1, shape =[self.prev_inp_size, size]), name = 'w')
        self.b = supervisor.variable(tf.constant(value=0.0, shape = [size]), name = 'b')
        self.has = []
        self.hbs = []
        for i in range(nbr_hinges):
            self.has.append(supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[size]), name = 'has'))
            self.hbs.append(supervisor.variable(tf.truncated_normal(stddev=0.1, shape=[size]), name = 'hbs'))

    def __call__(self, list_of_inputs):
        inp = list_of_inputs[0]
        x = tf.matmul(inp, self.w) + self.b
        linear_pieces = [tf.nn.relu(x)]
        for a,b in zip(self.has, self.hbs):
            # Adapation here
            ra = a
            rb = b
            piece = tf.multiply(ra, tf.nn.relu(tf.add(-x, rb)))
            linear_pieces.append(piece)
        output = tf.add_n(linear_pieces)
        return [output]

    def default_output(self):
        def1 = tf.placeholder_with_default(tf.fill([self.supervisor.batch_shape, self.size], 0.0),
                                           shape=[None, self.size])
        return [self.supervisor.store_op(def1, 'default_output')]