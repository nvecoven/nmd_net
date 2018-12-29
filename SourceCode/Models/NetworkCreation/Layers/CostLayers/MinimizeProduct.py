import tensorflow as tf

class MinimizeProduct():
    def __init__(self, supervisor):
        pass

    def get_all(self, input_list, lc):
        product = 1.0
        pred, out = self.get_pred(input_list)
        for inp in input_list:
            product = tf.multiply(product, inp)
        cost = tf.reduce_mean(product)
        return pred, out, cost

    def get_pred(self, input_list):
        pred = input_list[0]
        out = input_list[0]
        return pred, out