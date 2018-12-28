import tensorflow as tf

class MSE():
    def __init__(self, supervisor):
        pass

    def get_all(self, input_list, lc):
        pred, out = self.get_pred(input_list)
        cost = tf.reduce_mean(tf.multiply(tf.square(tf.subtract(out, input_list[1])), lc))
        return pred, out, cost

    def get_pred(self, input_list):
        pred = input_list[0]
        out = input_list[0]
        return pred, out