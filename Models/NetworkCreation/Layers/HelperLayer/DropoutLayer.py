import tensorflow as tf

# Takes one input which is the previous layer
class DropoutLayer():
    def __init__(self, supervisor, previous_size, size):
        self.prev_inp_size = previous_size[0]
        self.size = size
        self.supervisor = supervisor
        self.drop_prob = self.supervisor.op_dict['dropout'][0]

    def __call__(self, list_of_inputs):
        inp = list_of_inputs[0]
        out = tf.nn.dropout(inp, keep_prob=self.drop_prob)
        return [out]

    def default_output(self):
        def1 = tf.placeholder_with_default(tf.fill([self.supervisor.batch_shape, self.size], 0.0),
                                           shape=[None, self.size])
        return [self.supervisor.store_op(def1, 'default_output')]