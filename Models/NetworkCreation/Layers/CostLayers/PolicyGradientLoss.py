import tensorflow as tf

class PolicyGradientLoss():
    def __init__(self, supervisor):
        pass

    def get_all(self, input_list, lc):
        logp = input_list[0]
        old_logp = input_list[1]
        advantage = input_list[2]

        pred, out = self.get_pred(input_list)
        cost = -tf.reduce_mean(advantage * tf.exp(logp - old_logp))
        return pred, out, cost

    def get_pred(self, input_list):
        pred = input_list[3]
        out = input_list[3]
        return pred, out