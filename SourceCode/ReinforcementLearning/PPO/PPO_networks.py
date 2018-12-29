import tensorflow  as tf
from Models.NetworkCreation.Network import Network
import numpy as np
import pickle
import time

class PPO_networks(object):
    def __init__(self, obs_dim, action_dim, indexes, inpvecdim, policy_types, value_types, policy_sizes, value_sizes,
                 policy_params, value_params, policy_linker, value_linker,
                 init_policy_logvariance = -1.0, init_eta = 50.0, init_beta = 1.0,
                 kl_target = 0.003, value_lr = 6e-2, policy_lr = 2e-4, init_plr_mult = 1e-2, time_steps = 50,
                 file_name = None, cpu_only = False, save_policy_every = 10, name = 'default'):
        self.just_loaded = True
        if not file_name == None:
            self.model = Network(0, 0, 0, 0, 0, 0, 0, file_to_load_custom=file_name, cpu_only=cpu_only)
            self.to_pickle = pickle.load(open(file_name + '_networks_params', 'rb'))
        else:
            self.to_pickle = {}
            self.to_pickle['just_initialised'] = True
            self.to_pickle['indexes'] = indexes
            self.to_pickle['obs_dim'] = obs_dim
            self.to_pickle['action_dim'] = action_dim
            self.to_pickle['policy_update_counter'] = 0
            self.to_pickle['init_policlogvar'] = init_policy_logvariance
            self.to_pickle['eta'] = init_eta
            self.to_pickle['beta'] = init_beta
            self.to_pickle['kl_targ'] = kl_target
            self.to_pickle['value_lr'] = value_lr
            self.to_pickle['policy_lr'] = policy_lr
            self.to_pickle['lr_multiplier'] = init_plr_mult
            self.to_pickle['save_policy_every'] = save_policy_every
            self.to_pickle['name'] = name
            self.to_pickle['cpu_only'] = cpu_only

            policy_linker = [[[0,el[0],el[1]] if len(el) == 2 else el for el in el2] for el2 in policy_linker]
            value_linker = [[[5,el[0],el[1]] if len(el) == 2 else el for el in el2] for el2 in value_linker]
            value_types, value_linker, value_params, value_sizes = [value_types], [value_linker], [value_params], [
                value_sizes]
            policy_types, policy_linker, policy_params, policy_sizes = [policy_types], [policy_linker], [policy_params], [
                policy_sizes]

            net_layer_types = policy_types + \
                              [['InputLayer'],
                               ['InputLayer', 'AddLayer'],
                               ['NormalLogProbLayer', 'NormalLogProbLayer', 'KlLayer', 'EntropyLayer'],
                               ['SampleNormalLayer', 'ConcatLayer', 'ConcatLayer']] + \
                              value_types
            net_layer_sizes = policy_sizes + \
                              [[action_dim],
                               [action_dim, action_dim],
                               [1, 1, 1, 1],
                               [action_dim, action_dim * 2, 2]] + \
                              value_sizes
            net_layer_params = policy_params + \
                               [[[]],
                                [[], [self.to_pickle['init_policlogvar']]],
                                [[], [], [action_dim], [action_dim]],
                                [[], [], []]] + \
                               value_params
            # TODO : Maybe add previous action as input ????
            linker = policy_linker + \
                     [[[[0, len(net_layer_sizes[0]) - 1, 0]]],
                      [[[0, len(net_layer_sizes[0]) - 2, 0]], [[2, 0, 0]]],
                      [
                          [[1, 0, 0], [2, 1, 0], [-1, self.to_pickle['indexes']['act'][0], self.to_pickle['indexes']['act'][-1]]],
                          [[-1, self.to_pickle['indexes']['old_means'][0], self.to_pickle['indexes']['old_means'][-1]],
                           [-1, self.to_pickle['indexes']['old_vars'][0], self.to_pickle['indexes']['old_vars'][-1]],
                           [-1, self.to_pickle['indexes']['act'][0], self.to_pickle['indexes']['act'][-1]]],
                          [[-1, self.to_pickle['indexes']['old_vars'][0], self.to_pickle['indexes']['old_vars'][-1]], [2, 1, 0],
                           [-1, self.to_pickle['indexes']['old_means'][0], self.to_pickle['indexes']['old_means'][-1]], [1, 0, 0]],
                          [[2, 1, 0]]
                      ],
                      [[[1, 0, 0], [2, 1, 0]], [[1, 0, 0], [2, 1, 0]], [[3, 2, 0], [3, 3, 0]]]] + \
                     value_linker
            cost_type = ['PolicyGradientLoss', 'MinimizeProduct', 'ParametrizedHingeLoss','MSE','CopyOutput','CopyOutput']
            cost_params = [[],[],[self.to_pickle['kl_targ']],[],[],[]]
            cost_function_inputs = [[[3,0,0],[3,1,0],[-1,self.to_pickle['indexes']['advantages'][0],self.to_pickle['indexes']['advantages'][-1]],[4,0,0]],
                                    [[-1,self.to_pickle['indexes']['beta'][0],self.to_pickle['indexes']['beta'][-1]],[3,2,0]],
                                    [[-1,self.to_pickle['indexes']['eta'][0],self.to_pickle['indexes']['eta'][-1]],[3,2,0]],
                                    [[5,len(net_layer_sizes[5])-1,0],[-1,self.to_pickle['indexes']['disc_sum_rew'][0],self.to_pickle['indexes']['disc_sum_rew'][-1]]],
                                    [[4,1,0]],
                                    [[4,2,0]]]
            training_vars = [[0,], [5]]
            training_cost = [[0,1,2], [3]]
            variables_lists = None
            shuffle_intra = False
            self.model = Network(time_steps, inpvecdim, net_layer_sizes, net_layer_types, linker,
                                 cost_type, cost_function_inputs, shuffle_intra=shuffle_intra,
                                 tensorboard=False, cost_params=cost_params, net_params=net_layer_params,
                                 training_vars=training_vars, training_costs=training_cost, name='AC_PPO',
                                 variables_lists=variables_lists, dropout=0.5, cpu_only=cpu_only)

    def get_action(self, inp, state = None, elements_info = []):
        return self.model.take_step(inp, prev_state=state, prediction_nbr=0, elements=elements_info)

    def number_params(self):
        return self.model.get_number_of_params()

    def get_value(self, trajectories):
        value = self.model.predict(trajectories, prediction_nbr=3)[0]
        return value

    def update_value_function(self, trajectories, nbr_epochs, batch_size = 50):
        bs = min(batch_size, len(trajectories))
        self.model.fit_epoch(data=trajectories, nbr_epochs=nbr_epochs,
                             batch_size=bs, learning_rate=self.to_pickle['value_lr'], train_op_number=1)

    def update_policy(self, trajectories, nbr_epochs, batch_size = 50, check_early_stop = 5):
        #if self.just_loaded:
        #    self.just_loaded = False
        #    self.save_models('/tmp/' + str(self.to_pickle['name']))
        self.save_params('/tmp/' + self.to_pickle['name'])

        bs = min(batch_size, len(trajectories))
        # Mutate trajectories to include old means, old vars, eta and beta
        old_means_vars = self.model.predict(trajectories, prediction_nbr=4)[0]
        for t, tmv in zip(trajectories, old_means_vars):
            t[:,self.to_pickle['indexes']['eta'][0]] = self.to_pickle['eta']
            t[:,self.to_pickle['indexes']['beta'][0]] = self.to_pickle['beta']
            t[:, self.to_pickle['indexes']['old_means'][0]:self.to_pickle['indexes']['old_means'][-1]] = tmv[:, :self.to_pickle['action_dim']]
            t[:, self.to_pickle['indexes']['old_vars'][0]:self.to_pickle['indexes']['old_vars'][-1]] = tmv[:, self.to_pickle['action_dim']:]

        states_carries = None
        d = None
        for e in range(nbr_epochs):
            states_carries, d = self.model.fit_full_batch(data=trajectories, training_iterations=1,
                                                          batch_size=bs, learning_rate=self.to_pickle['policy_lr'] * self.to_pickle['lr_multiplier'],
                                                          train_op_number=0,
                                                          verbose=False, state_carries = states_carries, old_dict = d)
            # Could do some KL computation here for more efficient early stop ... is it owrth it ?
            # Compute KL and new entropy on FULL dataset
            # if e % check_early_stop == 0 and e > 0:
        kls_entropies = np.concatenate(self.model.predict(trajectories, prediction_nbr=5)[0])
        kl = np.mean(kls_entropies[:,0])
        entropy = np.mean(kls_entropies[:,1])
        if kl > self.to_pickle['kl_targ'] * 4 or np.isnan(kl):  # revert update
            print('EARLY STOP !')
            # self.model.close()
            print ('RESTORING PREVIOUS PARAMETERS !')
            # self.model = Network(0, 0, 0, 0, 0, 0,0, file_to_load_custom='/tmp/'+str(self.to_pickle['name']),
            #                     cpu_only=self.to_pickle['cpu_only'])
            self.restore_params('/tmp/'+self.to_pickle['name'])
            self.to_pickle['beta'] = np.minimum(35, 1.5 * self.to_pickle['beta'])  # max clip beta

        # TODO: too many "magic numbers" in next 8 lines of code, need to clean up
        if kl > self.to_pickle['kl_targ'] * 2:  # servo beta to reach D_KL target
            self.to_pickle['beta'] = np.minimum(35, 1.5 * self.to_pickle['beta'])  # max clip beta
            if self.to_pickle['beta'] > 30:
                self.to_pickle['lr_multiplier'] /= 1.5
        elif kl < self.to_pickle['kl_targ'] / 2:
            self.to_pickle['beta'] = np.maximum(1 / 35, self.to_pickle['beta'] / 1.5)  # min clip beta
            if self.to_pickle['beta'] < (1 / 30):
                self.to_pickle['lr_multiplier'] *= 1.5
        self.to_pickle['policy_update_counter'] += 1
        #if self.to_pickle['policy_update_counter'] % self.to_pickle['save_policy_every'] == 0 or self.to_pickle['policy_update_counter'] == 1\
        #        or self.just_loaded:
        #    print ('Saving temporarily policy !')
        #    self.just_loaded = False
        print ('KL :', kl)
        print ('Entropy : ', entropy)
        print ('Beta : ', self.to_pickle['beta'])
        print ('LR multiplier : ', self.to_pickle['lr_multiplier'])

    def close(self):
        self.model.close()

    def save_models(self, file_name):
        self.model.save_model_custom_path(file_name)
        pickle.dump(self.to_pickle, open(file_name + '_networks_params', 'wb'))

    def save_params(self, file_name):
        self.model.save_params_custom_path(file_name)
        pickle.dump(self.to_pickle, open(file_name + '_networks_params', 'wb'))

    def restore_params(self, file_name):
        self.model.load_params_custom_path(file_name)
        self.to_pickle = pickle.load(open(file_name + '_networks_params', 'rb'))
