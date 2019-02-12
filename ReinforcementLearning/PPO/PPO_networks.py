import tensorflow  as tf
from Models.NetworkCreation.Network import Network
import numpy as np
import pickle
import time

# Allows to define the actor and critic models to be trained, as well as the training procedure.
class PPO_networks(object):
    def __init__(self, obs_dim, action_dim, indexes, inpvecdim, policy_types, value_types, policy_sizes, value_sizes,
                 policy_params, value_params, policy_linker, value_linker,
                 init_policy_logvariance = -1.0, init_eta = 50.0, init_beta = 1.0,
                 kl_target = 0.003, value_lr = 6e-2, policy_lr = 2e-4, init_plr_mult = 1e-2, time_steps = 50,
                 file_name = None, cpu_only = False, save_policy_every = 10, name = 'default'):
        self.just_loaded = True
        # If a filename is given, then the corresponding network is loaded in memory
        if not file_name == None:
            self.model = Network(0, 0, 0, 0, 0, 0, 0, file_to_load_custom=file_name, cpu_only=cpu_only)
            self.to_pickle = pickle.load(open(file_name + '_networks_params', 'rb'))
        # Else the network is created from scratch
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

            # Each list inside the top level list defines a different part of the network.
            # net_layer_types[0] defines the actor
            # net_layer_types[1] defines the "mean" output of the actor
            # net_layer_types[2] defines the "logvariance" output of the actor
            # net_layers_types[3] defines the logvariance with the current params of the network,
            #                             the logvariance with the params of the network as were before the update,
            #                             the KL divergence between the current policy and the policy as before the update
            #                             the entropy of the current policy.
            # net_layers_types[4] allows to sample an action from the mean and logvar, gather the mean and
            #                            logvar and gather the entropy and KL divergence
            # net_layers_types[5] defines the critic
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
            # Define costs which will later be used to define the exact losses
            # CopyOutput is a trick to gather easily outputs of the network. This cost should never be part of a loss as
            # it wouldn't make sense.
            cost_type = ['PolicyGradientLoss', 'MinimizeProduct', 'ParametrizedHingeLoss','MSE','CopyOutput','CopyOutput']
            cost_params = [[],[],[self.to_pickle['kl_targ']],[],[],[]]
            cost_function_inputs = [[[3,0,0],[3,1,0],[-1,self.to_pickle['indexes']['advantages'][0],self.to_pickle['indexes']['advantages'][-1]],[4,0,0]],
                                    [[-1,self.to_pickle['indexes']['beta'][0],self.to_pickle['indexes']['beta'][-1]],[3,2,0]],
                                    [[-1,self.to_pickle['indexes']['eta'][0],self.to_pickle['indexes']['eta'][-1]],[3,2,0]],
                                    [[5,len(net_layer_sizes[5])-1,0],[-1,self.to_pickle['indexes']['disc_sum_rew'][0],self.to_pickle['indexes']['disc_sum_rew'][-1]]],
                                    [[4,1,0]],
                                    [[4,2,0]]]
            # Defines training operations. Training vars refer to a list of variables defined in the net_layer_types.
            # i.e. 0 -> actor and 5 -> critic
            # Training cost gives all the costs defined above that are averaged in a global loss function which will be
            # minimised by updating the corresponding training_vars
            # Hereunder, the actor's loss and critic's loss are defined.
            training_vars = [[0,], [5]]
            training_cost = [[0,1,2], [3]]
            variables_lists = None
            shuffle_intra = False
            # Finally, create the tensorflow model corresponding to all of the above.
            self.model = Network(time_steps, inpvecdim, net_layer_sizes, net_layer_types, linker,
                                 cost_type, cost_function_inputs, shuffle_intra=shuffle_intra,
                                 tensorboard=False, cost_params=cost_params, net_params=net_layer_params,
                                 training_vars=training_vars, training_costs=training_cost, name='AC_PPO',
                                 variables_lists=variables_lists, dropout=0.5, cpu_only=cpu_only)

    # Samples the policy. Elements_info allow to gather other informations in the network (such as
    # neuron's outputs in between others) is specified.
    def get_action(self, inp, state = None, elements_info = []):
        return self.model.take_step(inp, prev_state=state, prediction_nbr=0, elements=elements_info)

    # Number of variable in the actor-criticm odel
    def number_params(self):
        return self.model.get_number_of_params()

    # Critic's output
    def get_value(self, trajectories):
        value = self.model.predict(trajectories, prediction_nbr=3)[0]
        return value

    # Critic's update
    def update_value_function(self, trajectories, nbr_epochs, batch_size = 50):
        self.model.fit_epoch(data=trajectories, nbr_epochs=nbr_epochs,
                             batch_size=batch_size, learning_rate=self.to_pickle['value_lr'], train_op_number=1)

    # Actor's update.
    def update_policy(self, trajectories, nbr_epochs, batch_size = 50, check_early_stop = 5):
        self.save_params('/tmp/' + self.to_pickle['name'])

        old_means_vars = self.model.predict(trajectories, prediction_nbr=4)[0]
        for t, tmv in zip(trajectories, old_means_vars):
            t[:,self.to_pickle['indexes']['eta'][0]] = self.to_pickle['eta']
            t[:,self.to_pickle['indexes']['beta'][0]] = self.to_pickle['beta']
            t[:, self.to_pickle['indexes']['old_means'][0]:self.to_pickle['indexes']['old_means'][-1]] = tmv[:, :self.to_pickle['action_dim']]
            t[:, self.to_pickle['indexes']['old_vars'][0]:self.to_pickle['indexes']['old_vars'][-1]] = tmv[:, self.to_pickle['action_dim']:]

        self.model.fit_epoch(data=trajectories, nbr_epochs=nbr_epochs,
                             batch_size=batch_size,
                             learning_rate=self.to_pickle['policy_lr'] * self.to_pickle['lr_multiplier'],
                             train_op_number=0,
                             verbose=False)

        # Compute KL and new entropy on FULL dataset
        kls_entropies = np.concatenate(self.model.predict(trajectories, prediction_nbr=5)[0])
        kl = np.mean(kls_entropies[:,0])
        entropy = np.mean(kls_entropies[:,1])

        # Reverse update criterion

        # The following numbers and procedure has been taken from https://github.com/pat-coady/trpo and proved to
        # work well
        if kl > self.to_pickle['kl_targ'] * 4 or np.isnan(kl):  # revert update
            self.restore_params('/tmp/'+self.to_pickle['name'])
            self.to_pickle['beta'] = np.minimum(35, 1.5 * self.to_pickle['beta'])  # max clip beta

        if kl > self.to_pickle['kl_targ'] * 2:
            self.to_pickle['beta'] = np.minimum(35, 1.5 * self.to_pickle['beta'])
            if self.to_pickle['beta'] > 30:
                self.to_pickle['lr_multiplier'] /= 1.5
        elif kl < self.to_pickle['kl_targ'] / 2:
            self.to_pickle['beta'] = np.maximum(1 / 35, self.to_pickle['beta'] / 1.5)
            if self.to_pickle['beta'] < (1 / 30):
                self.to_pickle['lr_multiplier'] *= 1.5
        self.to_pickle['policy_update_counter'] += 1

        print ('KL :', kl)
        print ('Entropy : ', entropy)
        print ('Beta : ', self.to_pickle['beta'])
        print ('LR multiplier : ', self.to_pickle['lr_multiplier'])

    # Release tensorflow's metagraph memory
    def close(self):
        self.model.close()

    # Save tensorflow's metagraph as well as parameters. Should only be done once when
    # creating a new model
    def save_models(self, file_name):
        self.model.save_model_custom_path(file_name)
        pickle.dump(self.to_pickle, open(file_name + '_networks_params', 'wb'))

    # Saves the variables' value. Done after each iteration as it is not time consuming.
    # Allows to restart any model from where it was.
    def save_params(self, file_name):
        self.model.save_params_custom_path(file_name)
        pickle.dump(self.to_pickle, open(file_name + '_networks_params', 'wb'))

    # Reload the variables' value into the model.
    def restore_params(self, file_name):
        self.model.load_params_custom_path(file_name)
        self.to_pickle = pickle.load(open(file_name + '_networks_params', 'rb'))
