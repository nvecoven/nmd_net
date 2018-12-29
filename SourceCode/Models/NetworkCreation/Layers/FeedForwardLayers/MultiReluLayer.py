import tensorflow as tf

# Takes one input which is the previous layer
class MultiReluLayer():
    def __init__(self, supervisor, previous_size, size):
        self.prev_inp_size = previous_size
        self.size = size
        self.supervisor = supervisor
        self.ws = []
        for ps in previous_size:
            self.ws.append(supervisor.variable(tf.truncated_normal(stddev=0.1, shape =[ps, size]), name = 'w'))
        self.b = supervisor.variable(tf.constant(value=0.0, shape = [size]), name = 'b')

    def __call__(self, list_of_inputs):
        sum = self.b
        for inp, w in zip(list_of_inputs, self.ws):
            sum = sum + tf.matmul(inp, w)
        out = tf.nn.relu(sum)
        return [out]

    def default_output(self):
        def1 = tf.placeholder_with_default(tf.fill([self.supervisor.batch_shape, self.size], 0.0),
                                           shape=[None, self.size])
        return [self.supervisor.store_op(def1, 'default_output')]