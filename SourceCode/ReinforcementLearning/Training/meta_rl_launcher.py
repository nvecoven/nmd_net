#!/usr/bin/env python3
import os
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-tb", "--test_batch", type = int)
parser.add_argument("-bs", "--batch_size", type = int)
parser.add_argument("-o", "--offset", type = int)
parser.add_argument("-e", "--episodes", type = int, required = True)
parser.add_argument("-p", "--path", type = str)
parser.add_argument("-t", "--type", type = str)
parser.add_argument("-l", "--load", type = str)
parser.add_argument("-g", "--gpu", action="store_true")
parser.add_argument("-b", "--benchmark", type = int, required = True)

args = parser.parse_args()

train_steps = args.episodes

benchmark = args.benchmark

if args.offset:
    offset = args.offset
else:
    offset = 0

if args.gpu:
    cpu_only = False
else:
    cpu_only = True
    
if args.load:
    load = True
    load_type = args.load
else:
    load = False    
    
if args.test_batch:
    test_batch = args.test_batch
else:
    test_batch = 0

if args.batch_size:
    test_batch_size = args.batch_size
else:
    test_batch_size = 1

if args.type:
    type = args.type
else:
    type = 'nmdnet'


sys.path.append(os.getcwd())

from ReinforcementLearning.Environments.MovingTarget import MovingTarget
from ReinforcementLearning.Environments.MultipleReferences import MultipleReferences
from ReinforcementLearning.Environments.WindyReference import WindyReference
from ReinforcementLearning.PPO.PPO_manager import PPO_manager
import numpy as np

# Benchmark 1
if benchmark == 1:
    env = MovingTarget(fixed_offset=False, intra_variation=True, intra_variation_frequency=0.0, 
                       act_multiplicator=5.0, spike_reward=False)
    train_cut = 400

# Benchmark 2
elif benchmark == 2:
    env = WindyReference(control_speed = False, intra_var_percentage = 0.0, fixed_position = False,
                         fixed_reference = True, wind_half_cone=np.pi/5, wind_possible_dir=2*np.pi, 
                         wind_power = 1.0, max_steps = 4000)
    train_cut = 2000

# Benchmark 3
elif benchmark == 3:
    env = MultipleReferences(number_dots = 2, nbr_good = 1, fixed_good_refs = False, 
                             control_speed = False, fixed_references = True, 
                             fixed_position = False, stop_input = False, intra_var_pourcentage=0.0,
                             max_steps=5000, continuousmoving_references=False, 
                             targets_size_ratio = 1.0, bad_ref_rew = -50)
    train_cut = 2000

else:
    print("Please enter a benchmark number between 1 and 3")
    exit()

# Range of tests number to be carried out
tests = np.arange(test_batch * test_batch_size + offset, (test_batch+1) * test_batch_size)

for t in tests:
    print ("#!#!#!#!#!#!#!#!#! test ", str(t), "#!#!#!#!#!#!#!#!#!#!#!")

    name = "benchmark" + str(benchmark) + "_" + type + "_" + str(t)
    
    manager = PPO_manager(env, policy_epochs = 2, value_epochs = 1, value_replay_buffer_mult = 3,
                          policy_replay_buffer_mult = 1, name = name, check_early_stop=10, normalize_obs=False,
                          value_batch_size=250, trajectory_train_cut= train_cut)

    ############################### AVAL #########################################
    if type == 'nmdnet':
        if benchmark == 1:
            # Definition of the nmd net's recurrent layers
            nmd_part = ['GruLayer', 'ReluLayer', 'SplitLayer']
            nmd_sizes = [50, 20, [10, 10]]
            # params are added to allow modular layer creation
            nmd_params = [[], [''], []]
            # Definition of the nmd net's feed-forward layers
            ff_part = ['ParametrizedSaturatedReluLayer'] # PSR
            ff_sizes = [10]
            # params are added to allow modular layer creation
            ff_params = [['']]
            # Define the connections between layers
            beg = len(ff_part)+1
            nmd_linker = [[[beg, 0], [beg + 1, 0]], [[beg + 1, 0]], [[beg + 2, 0]]]
            nmd_o = len(ff_part) + len(nmd_part) + 1
            ff_linker = [[[0, 0], [nmd_o, 0]]]
        else:
            # Same architecture is used for benchmark 2 and benchmark 3
            # Definition of the recurrent part
            nmd_part = ['GruLayer', 'GruLayer', 'ReluLayer', 'SplitLayer']
            nmd_sizes = [100, 75, 45, [15, 15, 15]]
            nmd_params = [[], [], [''], []]
            # Definition of the feed-forward part
            ff_part = ['ParametrizedSaturatedReluLayer', 'ParametrizedSaturatedReluLayer']
            ff_sizes = [30, 10]
            ff_params = [[''], ['']]
            # Define the connections between layers
            beg = len(ff_part)+1
            nmd_linker = [[[beg, 0], [beg + 1, 0]], [[beg + 1, 0], [beg + 2, 0]], [[beg + 2, 0]], [[beg + 3, 0]]]
            nmd_o = len(ff_part) + len(nmd_part) + 1
            ff_linker = [[[0, 0], [nmd_o, 0]], [[1, 0], [nmd_o, 1]]]

        # Now merge the previous definition with the correct inputs, outputs and helper layers
        # Memory layer is used so that it remembers the output of a given layer from the previous time-step.
        # In this case it allows to remember the action taken as well as the reward obtained at the previous time-step and put them as 
        # inputs for the next time-step.
        # This trick is needed due to the order of definition of the different layers for the two nmd net's parts.
        policy_types = ['ConcatLayer'] + ff_part + ['ConcatLayer'] + nmd_part +['MemoryLayer', 'ParametricIdentity', 'ParametricIdentity']
        policy_sizes = [manager.to_pickle['nofb_obs_dim']] +  ff_sizes + [manager.to_pickle['obs_dim']] + nmd_sizes + \
                       [manager.to_pickle['nofb_obs_dim'], manager.to_pickle['act_dim'], manager.to_pickle['act_dim']]
        policy_params = [[]] + ff_params +[[]] + nmd_params + [[],[],[]]
        policy_linker = [manager.to_pickle['classic_observations']] + ff_linker +  [manager.to_pickle['fb_observations']+[[nmd_o+1,0]]] + nmd_linker + \
                        [manager.to_pickle['classic_observations'], [[len(ff_part),0],[nmd_o,len(nmd_sizes[-1])-1]], [[len(ff_part),0],[nmd_o,len(nmd_sizes[-1])-1]]]

        value_types = ['ConcatLayer'] + ff_part + ['ConcatLayer'] + nmd_part + ['MemoryLayer', 'ParametricIdentity']
        value_sizes = [manager.to_pickle['nofb_obs_dim']] + ff_sizes + [manager.to_pickle['obs_dim']] + nmd_sizes + \
                       [manager.to_pickle['nofb_obs_dim'], 1]
        value_params = [[]] + ff_params + [[]] + nmd_params + [[], []]
        value_linker = [manager.to_pickle['classic_observations']] + ff_linker + \
                       [manager.to_pickle['fb_observations'] + [[nmd_o + 1, 0]]] + nmd_linker + \
                       [manager.to_pickle['classic_observations'], [[len(ff_part), 0], [nmd_o, len(nmd_sizes[-1]) - 1]]]

    ############################### CLASSIC RECURRENT #######################################
    if type == 'recurrent':
        if benchmark == 1:
            types = ['GruLayer', 'SaturatedReluLayer', 'SaturatedReluLayer']
            sizes = [50, 20, 10]
            params = [[], [], []]
        else:
            types = ['GruLayer', 'GruLayer', 'SaturatedReluLayer', 'SaturatedReluLayer', 'SaturatedReluLayer']
            sizes = [100, 75, 45, 30, 10]
            params = [[], [], [], [], []]

        linker = [[[cnt, 0], [cnt + 1, 0]] if el == 'GruLayer' else [[cnt, 0]] for cnt, el in enumerate(types)]
        policy_types = ['ConcatLayer'] + types + ['LogitsLayer', 'LogitsLayer']
        policy_sizes = [manager.to_pickle['obs_dim']] + sizes + [manager.to_pickle['act_dim'], manager.to_pickle['act_dim']]
        policy_params = [[]] + params + [[],[]]
        policy_linker = [manager.to_pickle['full_obs']] + linker + [[[len(types),0]], [[len(types),0]]]

        value_types = ['ConcatLayer'] + types + ['LogitsLayer']
        value_sizes = [manager.to_pickle['obs_dim']] + sizes + [1]
        value_params = [[]] + params + [[]]
        value_linker = [manager.to_pickle['full_obs']] + linker + [[[len(types), 0]]]

    # Handle the case where it is the first time one runs a given test for a given benchmark and architecture.
    if not load:
        manager.build_networks(policy_types, value_types, policy_sizes, value_sizes,
                               policy_params, value_params, policy_linker, value_linker,
                               init_policy_logvariance = -1.0, init_eta = 50.0, init_beta = 1.0,
                               kl_target = 0.003, value_lr = 6e-3, policy_lr = 2e-4, init_plr_mult = 1.0,
                               time_steps = 1, cpu_only=cpu_only, save_policy_every=10)
    # Handle the case where one wants to load an already existing test. i.e. a test number with a type and a 
    # benchmark number that has already been ran.
    else:
        print ("LOADING MODEL ....")
        manager.init_networks(suffix = "_" + load_type, cpu_only = cpu_only, name = name)

    # Uncomment for pyplot visual example before training
    # manager.show_sample()
    manager.train(train_steps, batch_size=50, gamma = 0.998, lam=0.98, parallel=True)
    # Uncomment for pyplot visual example after training
    # manager.show_sample()
    manager.close()
