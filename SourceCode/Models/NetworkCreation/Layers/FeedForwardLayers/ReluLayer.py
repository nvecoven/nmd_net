import tensorflow as tf

# Takes one input which is the previous layer
class ReluLayer():
    def __init__(self, supervisor, previous_size, size, params):
        self.prev_inp_size = previous_size[0]
        self.nmd_type = params[0]
        self.size = size
        self.supervisor = supervisor
        self.w = supervisor.variable(tf.truncated_normal(stddev=0.1, shape =[self.prev_inp_size, size]), name = 'w')
        self.b = supervisor.variable(tf.constant(value=0.0, shape = [size]), name = 'b')

        if 'bistable' in self.nmd_type:
            self.a = supervisor.variable(tf.truncated_normal(stddev=0.1, shape =[size]), name = 'a')
            self.epsilon = supervisor.variable(tf.truncated_normal(stddev = 0.2, shape = [size]), name = 'epsilon')

    def __call__(self, list_of_inputs):
        inp = list_of_inputs[0]
        if 'bistable' in self.nmd_type:
            gprev = list_of_inputs[1]
        x = tf.nn.relu(tf.add(tf.matmul(inp, self.w), self.b))
        if 'bistable' in self.nmd_type:
            print ("Bistable neuron")
            multiplied_a = tf.maximum(0.0, self.a)
            rectified_epsilon = tf.clip_by_value(self.epsilon, clip_value_min=0.001, clip_value_max=1.0)
            fgt = tf.maximum(tf.minimum(gprev, -(gprev-multiplied_a)),
                             gprev - 2*multiplied_a)
            out = gprev + rectified_epsilon*(x - fgt)
        else:
            out = x
        return [out]

    def default_output(self):
        def1 = tf.placeholder_with_default(tf.fill([self.supervisor.batch_shape, self.size], 0.0),
                                           shape=[None, self.size])
        return [self.supervisor.store_op(def1, 'default_output')]