import copy

import numpy as np
import tensorflow as tf
import time

from .NetworkSupervisor import NetworkSupervisor
from ..DataFeeders.TrajectoriesSplitterNoGt import TrajectoriesSplitterNoGt

class Network(NetworkSupervisor):
    def create_network(self, time_steps, input_size, net_layers, net_layer_types, linker,
                       cost_types, cost_functions_inputs, training_vars = [[-1]],
                       training_costs = [0], shuffle_intra = False, net_params = None, cost_params = None,
                       variables_lists = None):
        if net_params == None:
            net_params = [[[] for el2 in el] for el in net_layers]

        if cost_params == None:
            cost_params = [[] for el in cost_types]

        if variables_lists == None:
            variables_lists = [[cnt for _ in el] for cnt, el in zip(range(len(net_layer_types)), net_layer_types)]

        self.to_pickle['input_size'] = input_size
        self.store_op(tf.placeholder(tf.float32), 'lr')
        inp = self.store_op(tf.placeholder(tf.float32, shape=[None, time_steps, input_size]), 'inputs')
        self.batch_shape = tf.shape(inp)[0]
        input_sequences_length = self.store_op(tf.placeholder(tf.float32, shape=[None, time_steps]),
                                               'input_sequences_length')
        # Define values required for training
        self.store_op(tf.placeholder(tf.bool), 'is_training')
        self.store_op(tf.placeholder(tf.float32), 'dropout')
        self.to_pickle['shuffle_intra'] = shuffle_intra
        if time_steps < 0:
            time_steps = 1
            self.to_pickle['time_steps'] = 1
            self.to_pickle['shuffle_intra'] = True

        cost_cells = []
        for cc, cp in zip(cost_types, cost_params):
            cost_cells.append(self.cost_layer(cc, cp, self))
        # Create variables
        nets = []
        net_number = 0
        with tf.name_scope('Variables'):
            for net, net_type, params, variables_list in zip(net_layers, net_layer_types, net_params, variables_lists):
                with tf.name_scope('Net_' + str(net_number)):
                    current_net_layers = []
                    for cnt in range(len(net)):
                        with tf.name_scope('Layer_' + str(cnt)):
                            layer_input_sizes = []
                            for link in linker[net_number][cnt]:
                                if link[0] >= 0:
                                    if isinstance(net_layers[link[0]][link[1]], list):
                                        layer_input_sizes.append(net_layers[link[0]][link[1]][link[2]])
                                    else:
                                        layer_input_sizes.append(net_layers[link[0]][link[1]])
                                else:
                                    layer_input_sizes.append(link[2]-link[1])
                        current_net_layers.append(self.layer(net_type[cnt], layer_input_sizes, net[cnt],
                                                             layer_params = params[cnt],
                                                             network_number=variables_list[cnt],
                                                             prefix='variables'+'net'+str(net_number)+'layer'+ str(cnt)+''))
                nets.append(current_net_layers)
                net_number += 1

        # Create default output for linking (corresponds to the initial value for each neuron's output)
        with tf.name_scope('Default_outputs'):
            default_nets_outputs = []
            for cnt1, net in enumerate(nets):
                with tf.name_scope('Net_'+str(cnt1)):
                    current_net_outputs = []
                    for kp, l in enumerate(net):
                        with tf.name_scope('Layer_'+str(kp)):
                            current_net_outputs.append(l.default_output())
                    default_nets_outputs.append(current_net_outputs)

        # First split the input and groundtruth such that we have a list of "time_step" lengths
        inp = tf.split(inp, time_steps, axis=1)
        lc = tf.split(input_sequences_length, time_steps, axis=1)

        costs = [[] for _ in cost_types]
        predictions = [[] for _ in cost_types]
        outputs = [[] for _ in cost_types]

        # Copy list of tensors to reuse for one step graph (used when interacting with the environment)
        nets_outputs = []
        for el in default_nets_outputs:
            cur = []
            for tl in el:
                cur.append(copy.copy(tl))
            nets_outputs.append(cur)

        for i in range(time_steps):
            with tf.name_scope('Timestep_' + str(i)):
                current_input = tf.squeeze(inp[i], axis=1)
                current_lc = lc[i]
                for net_number, net in enumerate(nets):
                    with tf.name_scope('Net_' + str(net_number)):
                        for l_number, l in enumerate(net):
                            with tf.name_scope('Layer_' + str(l_number)):
                                list_of_inputs = []
                                for to_append_input in linker[net_number][l_number]:
                                    if to_append_input[0] == -1:
                                        # Handle the case where we have inputs
                                        tmp = current_input[:, to_append_input[1]:to_append_input[2]]
                                        list_of_inputs.append(tmp)
                                    elif to_append_input[0] == -2:
                                        tmp = self.vars[to_append_input[1]]
                                        list_of_inputs.append(tmp)
                                    else:
                                        list_of_inputs.append(
                                            nets_outputs[to_append_input[0]][to_append_input[1]][to_append_input[2]])
                                layer_out = l(list_of_inputs)
                                for on, out in enumerate(layer_out):
                                    self.store_op(out, 'timestep' + str(i) + 'net' + str(net_number)
                                                  + 'layer' + str(l_number) + 'output' + str(on))
                                nets_outputs[net_number][l_number] = layer_out

                # Now that we have defined the network's structure, we need to define the cost and the outputs
                with tf.name_scope('Costs_and_preds'):
                    cost_nbr = 0
                    for function_inputs, cost_cell in zip(cost_functions_inputs, cost_cells):
                        cost_function_input_list = []
                        for to_append_input in function_inputs:
                            if to_append_input == []:
                                cost_function_input_list.append([])
                            elif to_append_input[0] == -1:
                                # Handle the case where we have inputs
                                cost_function_input_list.append(current_input[:, to_append_input[1]:to_append_input[2]])
                            elif to_append_input[0] == -2:
                                cost_function_input_list.append(self.vars[to_append_input[1]])
                            else:
                                cost_function_input_list.append(
                                    nets_outputs[to_append_input[0]][to_append_input[1]][to_append_input[2]])
                        p, o, c = cost_cell.get_all(cost_function_input_list, current_lc)
                        costs[cost_nbr].append(c)
                        predictions[cost_nbr].append(p)
                        outputs[cost_nbr].append(o)
                        cost_nbr += 1

        for net_number, net in enumerate(nets):
            for l_number, l in enumerate(net):
                for out in nets_outputs[net_number][l_number]:
                    self.store_op(out, 'previous_outputs')

        with tf.name_scope('Training'):
            for preds, outs, cs in zip(predictions, outputs, costs):
                self.store_op(tf.stack(preds, axis=1), 'predictions')
                self.store_op(tf.stack(outs, axis=1), 'final_output')
                cost = self.store_op(tf.stack(cs, axis=0), 'cutted_cost')
            cnt = 0
            for cur_vars, train_cost in zip(training_vars, training_costs):
                net_vars = []
                for net in cur_vars:
                    if net == -1:
                        for n in self.vars:
                            net_vars += self.vars[n]
                    else:
                        net_vars += self.vars[net]
                print ('Training vars for training op ', cnt)
                print (net_vars)
                optimiser = tf.train.AdamOptimizer(learning_rate=self.op_dict['lr'][-1])
                if isinstance(train_cost, list):
                    cur_cost = []
                    for tc in train_cost:
                        cur_cost.append(self.op_dict['cutted_cost'][tc])
                else:
                    cur_cost = [self.op_dict['cutted_cost'][train_cost]]
                print('Optimising cost : ', cur_cost)
                print('----------------')
                for x, c in enumerate(cur_cost):
                    if self.to_pickle['tensorboard']:
                        tf.summary.scalar('costs' + str(x), tf.reduce_mean(c))
                final_cost = self.store_op(tf.reduce_mean(tf.add_n(cur_cost)), 'final_costs')
                self.store_op(optimiser.minimize(final_cost,
                                                 var_list=net_vars), 'train')
                cnt += 1

        # Build the one-step graph
        with tf.name_scope('Onestep_graph'):
            print ('Building one step graph')
            inp = self.store_op(tf.placeholder(tf.float32, shape=[None, input_size]), 'inputs_onestep')

            nets_outputs = []
            for el in default_nets_outputs:
                cur = []
                for tl in el:
                    cur.append(copy.copy(tl))
                nets_outputs.append(cur)

            current_input = inp
            for net_number, net in enumerate(nets):
                for l_number, l in enumerate(net):
                    list_of_inputs = []
                    for to_append_input in linker[net_number][l_number]:
                        if to_append_input[0] == -1:
                            # Handle the case where we have inputs
                            list_of_inputs.append(current_input[:, to_append_input[1]:to_append_input[2]])
                        else:
                            list_of_inputs.append(nets_outputs[to_append_input[0]][to_append_input[1]][to_append_input[2]])
                    os_l_out = l(list_of_inputs)
                    nets_outputs[net_number][l_number] = os_l_out
                    for on, ol in enumerate(os_l_out):
                        self.store_op(ol, 'onestep' + 'net' + str(net_number)
                                      + 'layer' + str(l_number) + 'output' + str(on))

            # Now that we have defined everything, we need to define the outputs
            for function_inputs, cost_cell in zip(cost_functions_inputs, cost_cells):
                cost_function_input_list = []
                for to_append_input in function_inputs:
                    if to_append_input == []:
                        cost_function_input_list.append(current_input[:,0:1])
                    elif to_append_input[0] == -1:
                        # Handle the case where we have inputs
                        cost_function_input_list.append(current_input[:, to_append_input[1]:to_append_input[2]])
                    else:
                        cost_function_input_list.append(
                            nets_outputs[to_append_input[0]][to_append_input[1]][to_append_input[2]])
                p, o = cost_cell.get_pred(cost_function_input_list)
                self.store_op(p, 'prediction_onestep')
                self.store_op(o, 'output_onestep')

            for net_number, net in enumerate(nets):
                for l_number, l in enumerate(net):
                    for out in nets_outputs[net_number][l_number]:
                        self.store_op(out, 'previous_outputs_onestep')

        if self.to_pickle['tensorboard']:
            self.store_op(tf.summary.merge_all(), 'merged')

    # Uses the one step graph to feed previous state and current input to get current output and next states 
    # ( + any potential network output thanks to elements)
    def take_step(self, input, prev_state=None, prediction_nbr = 0, elements = []):
        d = {self.op_dict['inputs_onestep'][-1]: input, self.op_dict['dropout'][0]:1.0,
             self.op_dict['inputs'][-1]:np.zeros((len(input), self.to_pickle['time_steps'], self.to_pickle['input_size']))}
        if not prev_state == None:
            for s, obs in zip(self.op_dict['default_output'], prev_state):
                d[s] = obs
        if len(elements) == 0:
            prediction, state = self.sess.run([self.op_dict['prediction_onestep'][prediction_nbr],
                                               self.op_dict['previous_outputs_onestep']], feed_dict=d)
            values = []
        else:
            prediction, state, values = self.sess.run([self.op_dict['prediction_onestep'][prediction_nbr],
                                                       self.op_dict['previous_outputs_onestep']] +
                                                      [[self.op_dict['onestep' + 'net' + str(el[0]) +
                                                                    'layer' + str(el[1])+ 'output' + str(el[2])]
                                                       for el in elements]], feed_dict=d)
        return prediction, state, values

    def one_step(self, input, prev_state = None, prediction_nbr = 0):
        return self.take_step(input, prev_state, prediction_nbr)[:2]

    def one_step_with_values(self, input, elements, prev_state = None):
        return self.take_step(input, prev_state, elements = elements)

    # Feed forward pass to get all the states before splitting the trajectories to reduce the number of gradient updates
    def generate_samples_with_states(self, data, state_carries = None):
        time_steps = self.to_pickle['time_steps']
        if state_carries == None:
            states_carries = self.predict(data)[1]
        else:
            states_carries = state_carries
        batch_d = []
        batch_l = []
        previous_states = []
        default_state = [np.zeros_like(el) for el in states_carries[0][0]]
        for cnt, index in enumerate(range(len(data))):
            current_traj = data[index]
            traj_states = states_carries[index]
            splitted_current_traj = [current_traj[i:i + time_steps] for i in range(0, len(current_traj), time_steps)]
            nbr_cuts = len(splitted_current_traj)
            last_length = len(splitted_current_traj[-1])
            current_l = [[1.0] * len(el) for el in splitted_current_traj]
            if not last_length == time_steps:
                zeros_obs_to_add = [np.zeros_like(splitted_current_traj[0][0]).tolist()] * (time_steps - last_length)
                splitted_current_traj[-1] = np.concatenate([splitted_current_traj[-1], zeros_obs_to_add])
                current_l[-1] += [0.0] * (time_steps - last_length)
            batch_d.append(np.array(splitted_current_traj))
            batch_l.append(np.array(current_l))
            for k in range(nbr_cuts):
                if k == 0:
                    previous_states.append(default_state)
                else:
                    previous_states.append(traj_states[k * time_steps - 1])
        previous_states = np.array(previous_states)
        previous_states = [np.vstack(previous_states[:, i]) for i in range(previous_states.shape[1])]
        batch_d = np.array(np.concatenate(batch_d))
        batch_l = np.array(np.concatenate(batch_l))
        return previous_states, batch_d, batch_l

    # Fit the dataset for a given number of epochs
    def fit_epoch(self, data, nbr_epochs, train_op_number=0, batch_size=256,
                  verbose=False, learning_rate=1e-3, state_carries = None, processed_batches = None):
        batch_d, batch_l, previous_states = None, None, None
        if processed_batches == None:
            processed_batches = self.generate_samples_with_states(data, state_carries)
            previous_states, batch_d, batch_l = processed_batches
        training_op = self.op_dict['train'][train_op_number]
        batch_size = min(batch_size, batch_d.shape[0])
        if batch_size < 0:
            batch_size = batch_d.shape[0]
        for epoch in range(nbr_epochs):
            # Shuffle datas
            shuffeling_indexes = np.random.choice(batch_d.shape[0], batch_d.shape[0], replace=False)
            batch_d = batch_d[shuffeling_indexes]
            batch_l = batch_l[shuffeling_indexes]
            for i in range(len(previous_states)):
                previous_states[i] = previous_states[i][shuffeling_indexes]
            index = 0
            while not index >= batch_d.shape[0]:
                batch_indexes = np.arange(index, index+batch_size)
                index += batch_size
                d = {self.op_dict['inputs'][0]: batch_d[batch_indexes],
                     self.op_dict['inputs_onestep'][0]: batch_d[batch_indexes][0][:],
                     self.op_dict['input_sequences_length'][0]: batch_l[batch_indexes],
                     self.op_dict['is_training'][0]: True,
                     self.op_dict['dropout'][0]: self.to_pickle['dropout'],
                     self.op_dict['lr'][0]: learning_rate}
                for s, obs in zip(self.op_dict['default_output'], previous_states):
                    d[s] = obs[batch_indexes]
                if self.to_pickle['tensorboard']:
                    summary, _ = self.sess.run([self.op_dict['merged'][0], training_op], feed_dict=d)
                    self.summary_writer.add_summary(summary, self.to_pickle['training_iter'])
                else:
                    self.sess.run(training_op, feed_dict=d)
        return processed_batches
    
    # Fits the dataset for a given number of batches
    def fit_batch(self, data, training_iterations, train_op_number=0, batch_size=256,
                       verbose=False, learning_rate=1e-3, state_carries = None, processed_batches = None):
        batch_d, batch_l, previous_states = None, None, None
        if processed_batches == None:
            processed_batches = self.generate_samples_with_states(data, state_carries)
            previous_states, batch_d, batch_l = processed_batches
        training_op = self.op_dict['train'][train_op_number]
        batch_size = min(batch_size, batch_d.shape[0])
        if batch_size < 0:
            batch_size = batch_d.shape[0]
        for i in range(training_iterations):
            batch_indexes = np.random.choice(batch_d.shape[0], batch_size, replace = False)
            d = {self.op_dict['inputs'][0]: batch_d[batch_indexes],
                 self.op_dict['inputs_onestep'][0]: batch_d[batch_indexes][0][:],
                 self.op_dict['input_sequences_length'][0]: batch_l[batch_indexes],
                 self.op_dict['is_training'][0]: True,
                 self.op_dict['dropout'][0]: self.to_pickle['dropout'],
                 self.op_dict['lr'][0]: learning_rate}
            for s, obs in zip(self.op_dict['default_output'], previous_states):
                d[s] = obs[batch_indexes]
            if self.to_pickle['tensorboard']:
                summary, _ = self.sess.run([self.op_dict['merged'][0], training_op], feed_dict=d)
                self.summary_writer.add_summary(summary, self.to_pickle['training_iter'])
            else:
                self.sess.run(training_op, feed_dict=d)
        return processed_batches

    def get_score(self, data, carry_state=True, batch_size=50, train_last=False, shuffle_intra = False,
                  cost_nbr = 0):
        ts = TrajectoriesSplitterNoGt(data)
        ts.start_producing(batch_size=batch_size, time_steps=self.to_pickle['time_steps'],
                           ordered=True, nbr_of_batches=1, train_last=train_last,
                           shuffle_intra = shuffle_intra)
                           # shuffle_intra=self.to_pickle['shuffle_intra'])
        done = False
        scores = []
        while not done:
            x = ts.get_batch()
            if x == None:
                done = True
            else:
                full_batch_d, full_batch_l = x[0], x[1]
                previous_states = None
                for bd, bl in zip(full_batch_d, full_batch_l):
                    d = {self.op_dict['inputs'][0]: bd,
                         self.op_dict['input_sequences_length'][0]: bl,
                         self.op_dict['dropout'][0]: 1.0,
                         self.op_dict['is_training'][0]: False}
                    if not previous_states == None:
                        for s, obs in zip(self.op_dict['default_output'], previous_states):
                            d[s] = obs
                    previous_states, cost = self.sess.run([self.op_dict['previous_outputs'],
                                                           self.op_dict['final_costs'][cost_nbr]], feed_dict=d)
                    scores.append(cost)
        # handle lengths here ... not a simple_mean !
        return np.mean(scores)

    # Get a given output over a dataset (as well as the internal states values)
    def predict(self, datas, max_batch = 200, prediction_nbr=0):
        i = 0
        feature_size = len(datas[0][0])
        to_return = []
        full_states = []
        while i < len(datas):
            cur_datas = datas[i:i+max_batch]
            i += max_batch
            lengths = [len(el) for el in cur_datas]
            filler = [np.zeros((max(lengths)-l, feature_size)) for l in lengths]
            filled_cur_datas = [np.concatenate((el, el2)) for el, el2 in zip(cur_datas, filler)]
            prev_state = None
            transposed_predictions = []
            transposed_full_states = []
            for k in range(max(lengths)):
                results, prev_state = self.one_step(input=[el[k] for el in filled_cur_datas],
                                                    prev_state=prev_state, prediction_nbr=prediction_nbr)
                transposed_predictions.append(results)
                transposed_full_states.append(prev_state)

            tmp_arr = np.array(transposed_predictions)
            filled_current_states = []
            for sample in range(len(lengths)):
                filled_current_states.append([[el2[sample] for el2 in el] for el in transposed_full_states])
            filled_predictions = np.transpose(tmp_arr, [1,0,2])
            predictions = [el[:l] for el, l in zip(filled_predictions, lengths)]
            current_states = [el[:l] for el,l in zip(filled_current_states, lengths)]
            for pred_traj, cur_state in zip(predictions, current_states):
                to_return.append(pred_traj)
                full_states.append(cur_state)
        return to_return, full_states

    def get_var_by_index(self, index, name):
        return self.sess.run(self.op_dict['variables'+'net'+str(index[0])+'layer'+ str(index[1])+''+name])
