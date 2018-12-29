import numpy as np
from queue import Queue
from threading import Thread

class TrajectoriesSplitter():
    def __init__(self, data, answers, queue_max = 10, noise_type = 'uniform'):
        self.data = data
        self.nbr_trajectories = len(data)
        self.answers = answers
        self.data_filler = np.zeros_like(data[0][0]).tolist()
        self.answers_filler = np.zeros_like(answers[0][0]).tolist()
        # self.answers_filler = np.array(np.ones_like(answers[0][0])*1000).tolist()
        self.q = Queue(maxsize=queue_max)
        self.noise_type = noise_type

    def produce_batches(self, batch_size, time_steps, nbr_of_batches, steps_between_batches, ordered, train_last,
                        shuffle_intra):
        if steps_between_batches == None:
            steps_between_batches = time_steps
        if ordered:
            nbr_of_batches = -1
        start_batch = 0
        while nbr_of_batches != 0:
            batch_d = []
            batch_a = []
            batch_l = []
            if not ordered:
                trajectories_index = np.random.choice(self.nbr_trajectories, min(batch_size,self.nbr_trajectories),
                                                      replace = False)
            else:
                trajectories_index = range(start_batch, min(start_batch+batch_size, self.nbr_trajectories))
                start_batch += batch_size
            trajectories = [self.data[ind] for ind in trajectories_index]
            trajectories_answers = [self.answers[ind] for ind in trajectories_index]
            if shuffle_intra:
                t_ = []
                ta_ = []
                for t, ta in zip(trajectories, trajectories_answers):
                    p = np.random.choice(len(t), len(t), replace = False)
                    t_.append(t[p])
                    ta_.append(ta[p])
                trajectories, trajectories_answers = t_, ta_
            trajectories_lengths = [len(t) for t in trajectories]
            trajectories_remaining_samples = trajectories_lengths
            per_trajectory_index = [0 for t in trajectories]
            while not (np.sum(trajectories_remaining_samples) <= 0):
                current_d = [t[i:i+time_steps] for t,i in zip(trajectories, per_trajectory_index)]
                current_a = [ta[i:i+time_steps] for ta,i in zip(trajectories_answers, per_trajectory_index)]
                current_lengths = [len(t[i:i+time_steps]) for t,i in zip(trajectories, per_trajectory_index)]
                if not train_last:
                    cur_l = [[1.0]*l for l in current_lengths]
                else:
                    cur_l = [[0.0]*(l-1) + [1.0] for l in current_lengths]
                per_trajectory_index = [i+steps_between_batches for i in per_trajectory_index]
                trajectories_remaining_samples = [pl-min(ld,steps_between_batches) for pl, ld in
                                                  zip(trajectories_remaining_samples, current_lengths)]
                batch_d.append([np.array(el).tolist() + [self.data_filler]*(time_steps-l) for el,l in zip(current_d, current_lengths)])
                batch_a.append([np.array(el).tolist() + [self.answers_filler]*(time_steps-l) for el,l in zip(current_a, current_lengths)])
                batch_l.append([el + [0.0]*(time_steps-l) for el,l in zip(cur_l, current_lengths)])

            self.q.put([np.asarray(batch_d), np.asarray(batch_a), np.asarray(batch_l)], block = True)
            nbr_of_batches -= 1
            if start_batch >= self.nbr_trajectories:
                nbr_of_batches = 0
                self.q.put(None)

    def get_batch(self):
        tmp = self.q.get(block = True)
        if tmp == None:
            return None
        return tmp[0], tmp[1], tmp[2]

    def start_producing(self, batch_size, time_steps, nbr_of_batches, ordered = False, steps_between_batches = None,
                        train_last = False, shuffle_intra = False):
        self.t = Thread(target = self.produce_batches, args = (batch_size, time_steps, nbr_of_batches,
                                                               steps_between_batches, ordered, train_last,
                                                               shuffle_intra))
        self.t.start()

    def assert_stopped(self):
        self.t.join()