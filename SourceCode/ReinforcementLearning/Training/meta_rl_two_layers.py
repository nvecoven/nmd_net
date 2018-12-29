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

args = parser.parse_args()

train_steps = args.episodes

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
    type = 0

sys.path.append("/Users/nicolas/phd/svn_phd/phd/adaptative_nn/")

from ReinforcementLearning.Environments.GymPendulum import GymPendulum
from ReinforcementLearning.Environments.MyGymPendulum import MyGymPendulum
from ReinforcementLearning.Environments.IndependantBandits import IndependantBandits
from ReinforcementLearning.Environments.VaryingGaussian import VaryingGaussian
from ReinforcementLearning.Environments.MovingTarget import MovingTarget
from ReinforcementLearning.Environments.MapNavigator import MapNavigator
from ReinforcementLearning.Environments.MultipleReferences import MultipleReferences
from ReinforcementLearning.Environments.DeterministMultipleReferences import DeterministMultipleReferences

from ReinforcementLearning.Environments.WindyReference import WindyReference
from ReinforcementLearning.Environments.VisionReferenceFollowing import VisionReferenceFollowing
from ReinforcementLearning.PPO.PPO_manager import PPO_manager
import gym
import numpy as np

# Benchmark 1
#env = MovingTarget(fixed_offset=False, intra_variation=True, intra_variation_frequency=0.0, act_multiplicator=5.0, spike_reward=False)
#train_cut = 400

# Benchmark 2
env = MultipleReferences(number_dots = 2, nbr_good = 1, fixed_good_refs = False, control_speed = False,
                         fixed_references = True, fixed_position = False, stop_input = False, intra_var_pourcentage=0.0,
                         max_steps=5000, continuousmoving_references=False, targets_size_ratio = 1.0, bad_ref_rew = -50)
train_cut = 2000

# Benchmark 3
#env = WindyReference(control_speed = False, intra_var_percentage = 0.0, fixed_position = False,
#                     fixed_reference = True, wind_half_cone=np.pi/5, wind_possible_dir=2*np.pi, wind_power = 1.0, max_steps = 4000)
#train_cut = 2000

# env = MultipleReferences(number_dots = 2, nbr_good = 1, fixed_good_refs = False, control_speed = False,
#                  fixed_references = False, fixed_position = False, stop_input = False, intra_var_pourcentage=0.0)


# env = MapNavigator(fixed_reference=False, varying_ref_freq=0.025, split_reference=False, fixed_wind=True, varying_repulsion=True)

# env = DeterministMultipleReferences(number_dots = 2, nbr_good = 1, fixed_good_refs = False, control_speed = False,
#                          fixed_references = True, fixed_position = True, stop_input = False, intra_var_pourcentage=0.0,
#                          max_steps=400)

# env = VisionReferenceFollowing(0.0, 0.0, 0.0, relative_input = True, ref_speed = 1.0)

# env = GymPendulum()
# env = IndependantBandits(fixed_prob=False)
#test_batch = 4
#test_batch_size = 2
tests = np.arange(test_batch * test_batch_size + offset, (test_batch+1) * test_batch_size)
# load = False


#type = 'aval'
nmd_type = ''
# nmd_type = 'bistable'

# name = 'test2_multiplereferences_intravarying_0.05_goodrefs_' + type + nmd_type
# name = 'test2_multiplereferences_varyinggoodrefs_biggernet_' + type + nmd_type
# name = 'test1_multiplereferences_varyinggoodrefs_biggerff_' + type + nmd_type
# name = 'test2_new_performance_recurrence_bugged_' + type + nmd_type

# load = True
# name = 'test2_windyreference_saturatedrelu_varyingwind_probabilistic_' + type + nmd_type
# name = 'test2_windyreference_saturatedrelu_varyingwind_probabilistic_' + type + nmd_type

# name = 'test_deterministtargets_fixedposition_' + type + nmd_type
# name = 'test2_2000steps_securitycheck_2layers_continuousrefmove' + type + nmd_type
# name = 'test4_2000steps_securitycheck_2layers_oldsize' + type + nmd_type

# name = 'test9_multiplereferences_varyinggoodrefs_' + type + nmd_type
# load = True
# name = 'test4_multiplereferences_varyinggoodrefs_' + type + nmd_type

# name = 'indep_bandits_medium_nmd_smaller_' + type
# manager = PPO_manager(env, policy_epochs = 100, value_epochs = 100, value_replay_buffer_mult = 3,
#                       policy_replay_buffer_mult = 1, name = name, check_early_stop=10, normalize_obs=False,
#                       value_batch_size=100)

# name = 'test3_visionref_relative' + type + nmd_type
# name = 'article_navigation_test1' + type + nmd_type
for t in tests:
    print ("#!#!#!#!#!#!#!#!#! test ", str(t), "#!#!#!#!#!#!#!#!#!#!#!")
    # name = 'final_article_wdr_benchmark1_L1400_lprime400_E20000_type_' + type + '_A30_number_' + str(t)
    
    # name = 'final_article_wdr_benchmark2prime_L5000_lprime2000_E20000_type_' + type + '_A20_number_' + str(t) # Those gave decent preliminary results
    # name = 'final_article_wdr_benchmark2primeprime_L5000_lprime2000_E200000_type_' + type + '_A20_number_' + str(t)

    #name = 'final_article_wdr_benchmark3_onelayer_L4000_lprime2000_E100000_type_' + type + '_A20_number_' + str(t)
    #name = 'final_article_wdr_benchmark3_L4000_lprime2000_E100000_type_' + type + '_A20_number_' + str(t)
    
    # name = 'final_article_wdr_bignet_benchmark3_L4000_lprime2000_E100000_type_' + type + '_A20_number_' + str(t) # This is the good ones

    name = "benchmark2_halfsize_2ffnet_badrefrew" + type + "_" + str(t)#--> running
    
    #name = 'benchmark3_2fflayersnet_E100000_type_' + type + '_A20_number_' + str(t) # final results B3

    #name = 'test_gpu'
    manager = PPO_manager(env, policy_epochs = 2, value_epochs = 1, value_replay_buffer_mult = 3,
                          policy_replay_buffer_mult = 1, name = name, check_early_stop=10, normalize_obs=False,
                          value_batch_size=250, trajectory_train_cut= train_cut)

    ############################### DEFAULT FEEDFORWARD ##########################
    policy_types = ['ConcatLayer', 'ReluLayer', 'ReluLayer', 'ReluLayer', 'LogitsLayer', 'LogitsLayer']
    policy_sizes = [manager.to_pickle['obs_dim'], 1, 1, 1, manager.to_pickle['act_dim'], manager.to_pickle['act_dim']]
    # policy_params = [[],['bistable'],['bistable'],['bistable'],[],[]]
    policy_params = [[],[nmd_type],[nmd_type],[nmd_type],[],[]]
    policy_linker = [manager.to_pickle['full_obs'],
                     [[0, 0],[1,0]], [[1,0],[2,0]], [[2,0],[3,0]], [[3,0]], [[3,0]]]

    value_types = ['ConcatLayer', 'ReluLayer', 'ReluLayer', 'ReluLayer', 'LogitsLayer']
    value_sizes = [manager.to_pickle['obs_dim'], 50, 25, 7, 1]
    # value_params = [[],['bistable'],['bistable'],['bistable'],[]]
    value_params = [[],[nmd_type],[nmd_type],[nmd_type],[]]
    value_linker = [manager.to_pickle['full_obs'],
                    [[0, 0],[1,0]], [[1,0],[2,0]], [[2,0],[3,0]], [[3,0]]]
    ############################### AVAL #########################################
    # Bigger net has output of 50 ([25, 25]) from NMD net and has 50 adaptive relus
    if type == 'aval':
        nmd_part = ['GruLayer', 'GruLayer', 'ReluLayer', 'SplitLayer']
        # nmd_part = ['GruLayer', 'ReluLayer', 'SplitLayer']
        # ff_part = ['AdaptiveSaturatedReluLayer', 'AdaptiveSaturatedReluLayer']
        # ff_part = ['AdaptiveSaturatedReluLayer']
        ff_part = ['AdaptiveSaturatedReluLayer', 'AdaptiveSaturatedReluLayer']
        #ff_part = ['AdaptiveSaturatedReluLayer']
        # nmd_sizes = [100, 50, 40, [15, 15, 10]]
        # nmd_sizes = [100, 75, 50, [20, 20, 10]]

        # nmd_sizes = [100, 100, 20, [10, 10]] # GOOD ? ?? for b2

        #nmd_sizes = [500, 500, 300, [150, 150]]

        nmd_sizes = [100, 75, 45, [15, 15, 15]] # GOOD FOR WIND !
        # nmd_sizes = [100, 100, 45, [15, 15, 15]]

        # nmd_sizes = [100, 100, 20, [10, 10]]
        ff_sizes = [30, 10]
        # ff_sizes = [10] # GOOD ? ?? for b2
        nmd_params = [[], [], [''], []]
        # nmd_params = [[], [''], []]
        ff_params = [[nmd_type], [nmd_type]]
        # ff_params = [[nmd_type]]
        beg = len(ff_part)+1
        # nmd_linker = [[[beg, 0], [beg + 1, 0]], [[beg + 1, 0]], [[beg + 2, 0]]]
        nmd_linker = [[[beg, 0], [beg + 1, 0]], [[beg + 1, 0], [beg + 2, 0]], [[beg + 2, 0]], [[beg + 3, 0]]]
        nmd_o = len(ff_part) + len(nmd_part) + 1
        ff_linker = [[[0, 0], [nmd_o, 0]], [[1, 0], [nmd_o, 1]]]
        # ff_linker = [[[0, 0], [nmd_o, 0], [1, 0]]]
        # ff_linker = [[[0, 0], [nmd_o, 0]]]

        policy_types = ['ConcatLayer'] + ff_part + ['ConcatLayer'] + nmd_part +['MemoryLayer', 'AdaptiveLogitsLayer2', 'AdaptiveLogitsLayer2']
        policy_sizes = [manager.to_pickle['nofb_obs_dim']] +  ff_sizes + [manager.to_pickle['obs_dim']] + nmd_sizes + \
                       [manager.to_pickle['nofb_obs_dim'], manager.to_pickle['act_dim'], manager.to_pickle['act_dim']]
        policy_params = [[]] + ff_params +[[]] + nmd_params + [[],[],[]]
        policy_linker = [manager.to_pickle['classic_observations']] + ff_linker +  [manager.to_pickle['fb_observations']+[[nmd_o+1,0]]] + nmd_linker + \
                        [manager.to_pickle['classic_observations'], [[len(ff_part),0],[nmd_o,len(nmd_sizes[-1])-1]], [[len(ff_part),0],[nmd_o,len(nmd_sizes[-1])-1]]]

        print (policy_types)
        print (policy_sizes)
        print (policy_linker)

        value_types = ['ConcatLayer'] + ff_part + ['ConcatLayer'] + nmd_part + ['MemoryLayer', 'AdaptiveLogitsLayer2']
        value_sizes = [manager.to_pickle['nofb_obs_dim']] + ff_sizes + [manager.to_pickle['obs_dim']] + nmd_sizes + \
                       [manager.to_pickle['nofb_obs_dim'], 1]
        value_params = [[]] + ff_params + [[]] + nmd_params + [[], []]
        value_linker = [manager.to_pickle['classic_observations']] + ff_linker + \
                       [manager.to_pickle['fb_observations'] + [[nmd_o + 1, 0]]] + nmd_linker + \
                       [manager.to_pickle['classic_observations'], [[len(ff_part), 0], [nmd_o, len(nmd_sizes[-1]) - 1]]]

    ############################### AVAL #########################################
    # Bigger net has output of 50 ([25, 25]) from NMD net and has 50 adaptive relus
    if type == 'aval_onelayer':
        nmd_part = ['GruLayer', 'ReluLayer', 'SplitLayer']
        # ff_part = ['AdaptiveSaturatedReluLayer', 'AdaptiveSaturatedReluLayer']
        ff_part = ['AdaptiveSaturatedReluLayer']
        # ff_part = ['AdaptiveSaturatedReluLayer', 'AdaptiveSaturatedReluLayer']
        # ff_part = ['AdaptiveReluLayer']
        # nmd_sizes = [100, 50, 40, [15, 15, 10]]
        # nmd_sizes = [100, 75, 50, [20, 20, 10]]
        nmd_sizes = [50, 20, [10, 10]]
        # ff_sizes = [30, 10]
        ff_sizes = [10]
        nmd_params = [[], [], []]
        # ff_params = [[nmd_type], [nmd_type]]
        ff_params = [[nmd_type]]
        beg = len(ff_part)+1
        nmd_linker = [[[beg,0],[beg+1,0]], [[beg+1,0]], [[beg+2,0]]]
        nmd_o = len(ff_part) + len(nmd_part) + 1
        # ff_linker = [[[0, 0], [nmd_o, 0]], [[1, 0], [nmd_o, 1]]]
        # ff_linker = [[[0, 0], [nmd_o, 0], [1, 0]]]
        ff_linker = [[[0, 0], [nmd_o, 0]]]

        policy_types = ['ConcatLayer'] + ff_part + ['ConcatLayer'] + nmd_part +['MemoryLayer', 'AdaptiveLogitsLayer2', 'AdaptiveLogitsLayer2']
        policy_sizes = [manager.to_pickle['nofb_obs_dim']] +  ff_sizes + [manager.to_pickle['obs_dim']] + nmd_sizes + \
                       [manager.to_pickle['nofb_obs_dim'], manager.to_pickle['act_dim'], manager.to_pickle['act_dim']]
        policy_params = [[]] + ff_params +[[]] + nmd_params + [[],[],[]]
        policy_linker = [manager.to_pickle['classic_observations']] + ff_linker +  [manager.to_pickle['fb_observations']+[[nmd_o+1,0]]] + nmd_linker + \
                        [manager.to_pickle['classic_observations'], [[len(ff_part),0],[nmd_o,len(nmd_sizes[-1])-1]], [[len(ff_part),0],[nmd_o,len(nmd_sizes[-1])-1]]]

        print (policy_types)
        print (policy_sizes)
        print (policy_linker)

        value_types = ['ConcatLayer'] + ff_part + ['ConcatLayer'] + nmd_part + ['MemoryLayer', 'AdaptiveLogitsLayer2']
        value_sizes = [manager.to_pickle['nofb_obs_dim']] + ff_sizes + [manager.to_pickle['obs_dim']] + nmd_sizes + \
                       [manager.to_pickle['nofb_obs_dim'], 1]
        value_params = [[]] + ff_params + [[]] + nmd_params + [[], []]
        value_linker = [manager.to_pickle['classic_observations']] + ff_linker + \
                       [manager.to_pickle['fb_observations'] + [[nmd_o + 1, 0]]] + nmd_linker + \
                       [manager.to_pickle['classic_observations'], [[len(ff_part), 0], [nmd_o, len(nmd_sizes[-1]) - 1]]]

    ############################### AMONT #########################################
    if type == 'amont':
        policy_types = ['ConcatLayer', 'GruLayer', 'GruLayer', 'ReluLayer', 'SplitLayer', 'ConcatLayer', 'AdaptiveSaturatedReluLayer',
                        'AdaptiveLogitsLayer2','AdaptiveLogitsLayer2']
        policy_sizes = [manager.to_pickle['obs_dim'], 100, 100, 20, [10, 10], manager.to_pickle['nofb_obs_dim'],
                        10, manager.to_pickle['act_dim'],manager.to_pickle['act_dim']]
        policy_params = [[],[],[],[],[],[],[nmd_type],[],[]]
        policy_linker = [manager.to_pickle['full_obs'], [[0,0],[1,0]],[[1,0],[2,0]], [[2,0]], [[3,0]],
                         manager.to_pickle['classic_observations'], [[5,0],[4,0]], [[6,0],[4,1]],[[6,0],[4,1]]]

        value_types = ['ConcatLayer', 'GruLayer', 'GruLayer', 'ReluLayer', 'SplitLayer', 'ConcatLayer', 'AdaptiveSaturatedReluLayer',
                        'AdaptiveLogitsLayer2']
        value_sizes = [manager.to_pickle['obs_dim'], 100, 100, 20, [10, 10], manager.to_pickle['nofb_obs_dim'],
                        10, 1]
        value_params = [[], [], [], [], [], [], [nmd_type], []]
        value_linker = [manager.to_pickle['full_obs'], [[0, 0], [1, 0]], [[1,0],[2,0]], [[2, 0]], [[3, 0]],
                         manager.to_pickle['classic_observations'], [[5, 0], [4, 0]], [[6, 0], [4, 1]]]

    ##############################3 CLASSIC RECURRENT #######################################
    if type == 'recurrent':
        types = ['GruLayer', 'GruLayer', 'SaturatedReluLayer', 'SaturatedReluLayer', 'SaturatedReluLayer']

        # types = ['GruLayer', 'GruLayer', 'ReluLayer', 'ReluLayer']

        # types = ['GruLayer', 'GruLayer', 'SaturatedReluLayer', 'SaturatedReluLayer']
        # types = ['GruLayer', 'ReluLayer', 'ReluLayer']
        sizes = [1, 1, 1, 1, 1]

        # sizes = [100, 100, 20, 10]

        # sizes = [50, 20, 15]

        # sizes = [75, 50, 20, 20]
        # sizes = [100, 100, 20, 10]
        params = [[], [], [], [], []]

        # params = [[], [], [''], ['']]

        #params = [[], [''], ['']]

        linker = [[[cnt, 0], [cnt + 1, 0]] if el == 'GruLayer' else [[cnt, 0]] for cnt, el in enumerate(types)]
        policy_types = ['ConcatLayer'] + types + ['LogitsLayer', 'LogitsLayer']
        policy_sizes = [manager.to_pickle['obs_dim']] + sizes + [manager.to_pickle['act_dim'], manager.to_pickle['act_dim']]
        policy_params = [[]] + params + [[],[]]
        policy_linker = [manager.to_pickle['full_obs']] + linker + [[[len(types),0]], [[len(types),0]]]

        print (policy_types)
        print (policy_sizes)
        print (policy_linker)

        value_types = ['ConcatLayer'] + types + ['LogitsLayer']
        value_sizes = [manager.to_pickle['obs_dim']] + sizes + [1]
        value_params = [[]] + params + [[]]
        value_linker = [manager.to_pickle['full_obs']] + linker + [[[len(types), 0]]]

    ############################## Default single layer network which has worked #############
    if type == 'recurrent_one_layer':
        policy_types = ['ConcatLayer', 'GruLayer', 'ReluLayer', 'LogitsLayer', 'LogitsLayer']
        policy_sizes = [manager.to_pickle['obs_dim'], 50, 25, manager.to_pickle['act_dim'],manager.to_pickle['act_dim']]
        policy_params = [[], [], [], [], []]
        policy_linker = [manager.to_pickle['full_obs'],[[0, 0], [1, 0]], [[1, 0]], [[2,0]], [[2,0]]]

        value_types = ['ConcatLayer', 'GruLayer', 'ReluLayer', 'LogitsLayer']
        value_sizes = [manager.to_pickle['obs_dim'], 50, 25, 1]
        value_params = [[], [], [], []]
        value_linker = [manager.to_pickle['full_obs'],
                        [[0, 0], [1, 0]], [[1, 0]], [[2,0]]]



    if not load:
        manager.build_networks(policy_types, value_types, policy_sizes, value_sizes,
                               policy_params, value_params, policy_linker, value_linker,
                               init_policy_logvariance = -1.0, init_eta = 50.0, init_beta = 1.0,
                               kl_target = 0.003, value_lr = 6e-3, policy_lr = 2e-4, init_plr_mult = 1.0,
                               time_steps = 1, cpu_only=cpu_only, save_policy_every=10)
    else:
        print ("LOADING MODEL ....")
        #manager = PPO_manager(env, policy_epochs = 100, value_epochs = 100, value_replay_buffer_mult = 3, load = True,
        #                      policy_replay_buffer_mult = 1, name = name, check_early_stop=10, normalize_obs=False,
        #                      suffix = '_checkpoint', cpu_only=cpu_only)
        manager.init_networks(suffix = "_" + load_type, cpu_only = cpu_only, name = name)
    # manager.show_sample()
    # manager.show_sample()

    for t in range(1):
        # manager.show_sample()
        manager.train(train_steps, batch_size=50, gamma = 0.998, lam=0.98, parallel=True)
        print ('###########################', name, ' with ', manager.get_number_params(), ' parameters ###############################')
    manager.close()
