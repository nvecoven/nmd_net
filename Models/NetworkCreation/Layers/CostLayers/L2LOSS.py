import tensorflow as tf

class L2LOSS():
    def __init__(self, supervisor, params):
        self.supervisor = supervisor
        self.L2AMPLITUDE = params[0]
        pass

    def get_all(self, input_list, lc):
        # Flatten if multiple lists of variables
        to_regularise = [item for sublist in input_list for item in sublist]
        losses = [tf.nn.l2_loss(t) * self.L2AMPLITUDE for t in to_regularise]
        pred, out = self.get_pred(input_list)
        cost = tf.add_n(losses)
        return pred, out, cost

    def get_pred(self, input_list):
        pred = input_list[0]
        out = input_list[0]
        return pred, out