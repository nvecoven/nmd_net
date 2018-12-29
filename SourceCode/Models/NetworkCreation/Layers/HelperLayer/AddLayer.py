import tensorflow as tf

# Takes one input which is the previous layer
class AddLayer():
    def __init__(self, supervisor, previous_size, size, params):
        self.value = params[0]
        self.prev_inp_size = previous_size[0]
        self.size = size
        self.supervisor = supervisor

    def __call__(self, list_of_inputs):
        inp = list_of_inputs[0]
        output = tf.add(inp, self.value)
        return [output]

    def default_output(self):
        def1 = tf.placeholder_with_default(tf.fill([self.supervisor.batch_shape, self.size], 0.0),
                                           shape=[None, self.size])
        return [self.supervisor.store_op(def1, 'default_output')]