import tensorflow as tf

class SplitLayer():
    def __init__(self, supervisor, previous_size, size):
        self.size = size
        self.supervisor = supervisor

    def __call__(self, list_of_inputs):
        inp = list_of_inputs[0]
        cnt = 0
        out = []
        for s in self.size:
            out.append(inp[:, cnt:cnt+s])
            cnt += s
        return out

    def default_output(self):
        default = []
        for s in self.size:
            defx = tf.placeholder_with_default(tf.fill([self.supervisor.batch_shape, s], 0.0),
                                               shape=[None, s])
            default.append(self.supervisor.store_op(defx, 'default_output'))
        return default