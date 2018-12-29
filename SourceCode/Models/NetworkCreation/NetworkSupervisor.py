import pickle

import numpy as np
import tensorflow as tf
import os
import importlib

class NetworkSupervisor():
    def __init__(self, time_steps, input_size, net_layers, net_layer_types, linker,
                 cost_types, cost_functions_inputs, training_vars = [[-1]],
                 training_costs = [0],
                 file_to_load = None, tensorboard = False, shuffle_intra = False, ptype = 'regression',
                 dropout = 0.5, name = 'my_network', net_params = None, cost_params = None, variables_lists = None,
                 file_to_load_prefix = '', file_to_load_custom = None, cpu_only = False):
        np.random.seed(1)
        if not file_to_load == None:
            self.load_model(file_to_load, relative_prefix=file_to_load_prefix, cpu_only = cpu_only)
        elif not file_to_load_custom == None:
            self.load_model_custom_path(file_to_load_custom, cpu_only=cpu_only)
        else:
            self.to_pickle = {}
            self.op_dict = {}
            self.to_pickle['tensorboard'] = tensorboard
            self.to_pickle['ptype'] = ptype
            self.to_pickle['time_steps'] = time_steps
            self.to_pickle['dropout'] = dropout
            self.to_pickle['shuffle_intra'] = shuffle_intra
            self.to_pickle['name'] = name
            self.vars = {}
            self.to_regularise_vars = []
            self.create_network(time_steps, input_size, net_layers, net_layer_types, linker,
                                cost_types, cost_functions_inputs,
                                training_vars = training_vars, training_costs = training_costs,
                                shuffle_intra = shuffle_intra, net_params = net_params, cost_params = cost_params,
                                variables_lists = variables_lists)
            if cpu_only:
                config = tf.ConfigProto(
                    device_count={'GPU': 0}
                )
                self.sess = tf.Session(config=config)
            else:
                self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self.to_pickle['training_iter'] = 0
            self.net_number = 0
            self.current_prefix = ''
        if self.to_pickle['tensorboard']:
            self.summary_writer = tf.summary.FileWriter('/tmp/' + self.to_pickle['name'], self.sess.graph)

    def create_network(self, time_steps, input_size, net_layers, net_layer_types, linker,
                       cost_types, cost_functions_inputs, training_vars = [[-1]],
                       training_costs = [0], shuffle_intra = False, net_params = None, cost_params = None,
                       variables_lists = None):
        print ('Needs to be implemented')

    def variable(self, t, var_list = None, trainable = True, name = 'default_var'):
        if var_list == None:
            list_name = self.net_number
        else:
            list_name = var_list
        tmp = tf.Variable(t, trainable=trainable)
        if list_name in self.vars:
            self.vars[list_name].append(tmp)
        else:
            self.vars[list_name] = [tmp]
        return self.store_op(tmp, self.current_prefix + name)

    def regularised_variable(self, t, trainable = True):
        var = self.variable(t, trainable)
        self.to_regularise_vars.append(var)
        return var

    def save_model(self, file_path, relative_prefix=''):
        file_path = relative_prefix + 'Models/saved_models/' + file_path
        saver = tf.train.Saver()
        saver.save(self.sess, file_path)
        pickle.dump(self.to_pickle, open(file_path + '.pickle', 'wb'))

    def load_model(self, file_path, relative_prefix='', cpu_only = False):
        file_path = relative_prefix + 'Models/saved_models/' + file_path
        if not cpu_only:
            self.sess = tf.Session()
        else:
            config = tf.ConfigProto(
                device_count = {'GPU': 0}
            )
            self.sess = tf.Session(config = config)
        saver = tf.train.import_meta_graph(file_path + '.meta')
        saver.restore(self.sess, file_path)
        self.to_pickle = pickle.load(open(file_path + '.pickle', 'rb'))

        self.op_dict = {}
        for op in tf.get_default_graph().get_all_collection_keys():
            self.op_dict[op] = tf.get_collection(op)

    def save_model_custom_path(self, file_path):
        saver = tf.train.Saver()
        saver.save(self.sess, file_path)
        pickle.dump(self.to_pickle, open(file_path + '.pickle', 'wb'))

    def load_model_custom_path(self, file_path, cpu_only = False):
        if not cpu_only:
            self.sess = tf.Session()
        else:
            config = tf.ConfigProto(
                device_count = {'GPU': 0}
            )
            self.sess = tf.Session(config = config)
        saver = tf.train.import_meta_graph(file_path + '.meta')
        saver.restore(self.sess, file_path)
        self.to_pickle = pickle.load(open(file_path + '.pickle', 'rb'))

        self.op_dict = {}
        for op in tf.get_default_graph().get_all_collection_keys():
            self.op_dict[op] = tf.get_collection(op)
        
    def save_params_custom_path(self, file_path):
        print ("Writing only params...")
        saver = tf.train.Saver()
        saver.save(self.sess, file_path, write_meta_graph = False)
        pickle.dump(self.to_pickle, open(file_path + '.pickle', 'wb'))
        print ("Writing done !")

    def load_params_custom_path(self, file_path, cpu_only = False):
        print ("Restoring only params !")
        saver = tf.train.Saver()
        saver.restore(self.sess, file_path)
        self.to_pickle = pickle.load(open(file_path + '.pickle', 'rb'))
        print ("Restore done !")

    def store_op(self, op, name):
        if name in self.op_dict:
            self.op_dict[name].append(op)
        else:
            self.op_dict[name] = [op]
        tf.add_to_collection(name, op)
        return op

    def close(self):
        self.sess.close()
        tf.reset_default_graph()

    def reset_network(self):
        self.sess.run(tf.global_variables_initializer())

    def get_var_value(self, list_name):
        return self.sess.run(self.vars[list_name])

    def get_op_value(self, el):
        return self.sess.run(self.op_dict[el])

    def get_ops(self, el):
        return self.op_dict[el]

    def get_sess(self):
        return self.sess

    def layer(self, type, previous_size, size, layer_params, network_number = 0, prefix = ''):
        self.net_number = network_number
        self.current_prefix = prefix
        for dirname, dirnames, filenames in os.walk('./Models/NetworkCreation/Layers'):
            for filename in filenames:
                dirname = dirname.replace('.', '').replace('/', '.')
                dirname = dirname[1:]
                try:
                    mod = importlib.import_module(dirname + '.' + type)
                    obj = getattr(mod, type)
                    print ('LAYER ', type, ' CONSTRUCTED')
                    if layer_params == []:
                        return obj(self, previous_size, size)
                    else:
                        return obj(self, previous_size, size, layer_params)
                except Exception as err:
                    pass
        print ('NO ', type , ' LAYER FOUND !')
        return

    def cost_layer(self, type, param, supervisor):
        for dirname, dirnames, filenames in os.walk('./Models/NetworkCreation/Layers/CostLayers'):
            for filename in filenames:
                dirname = dirname.replace('.', '').replace('/', '.')
                dirname = dirname[1:]
                try:
                    mod = importlib.import_module(dirname + '.' + type)
                    obj = getattr(mod, type)
                    if param == []:
                        return obj(supervisor)
                    else:
                        return obj(supervisor, param)
                except Exception as err:
                    print(err)
                    return

    def get_number_of_params(self):
        return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
