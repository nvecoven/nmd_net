import tensorflow as tf

class ParametrizedHingeLoss():
    def __init__(self, supervisor, params):
        self.kl_targ = params[0]

    def get_all(self, input_list, lc):
        pred, out = self.get_pred(input_list)
        multiplier = input_list[0]
        kl = input_list[1]
        cost = tf.reduce_mean(multiplier * tf.square(tf.maximum(0.0, kl - 2.0*self.kl_targ)))
        return pred, out, cost

    def get_pred(self, input_list):
        pred = input_list[0]
        out = input_list[0]
        return pred, out