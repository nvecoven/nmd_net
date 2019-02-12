from ReinforcementLearning.PPO.PPO_networks import PPO_networks
import numpy as np
import scipy
import scipy.signal
import time
import pickle
import copy
import resource
from numpy.polynomial.polynomial import polyval

# Performs the meta-RL algorithm
class PPO_manager():
    def __init__(self, env, policy_epochs = 50, value_epochs = 20, check_early_stop = 20, value_replay_buffer_mult = 3,
                 policy_replay_buffer_mult = 1, normalize_obs = True, name = '', load = False, suffix = '', model_suffix = '_initialmodel', cpu_only = False,
                 value_batch_size = 50, trajectory_train_cut = 2000):
        self.env = env
        # In case the test has never been ran
        if not load:
            self.to_pickle = {}
            self.to_pickle['episode'] = 0
            self.to_pickle['trajectory_train_cut'] = trajectory_train_cut # L' hyper-param
            # state-space dimension + action-space dimension + 1 -> dimension of the nmd net's recurrent part input
            self.to_pickle['obs_dim'] = env.obs_dim
            self.to_pickle['act_dim'] = env.act_dim # action-space dimension
            self.to_pickle['nofb_obs_dim'] = env.nofb_obs_dim # nmd net's feed-forward part's input dimension
            self.to_pickle['value_epochs'] = value_epochs
            self.to_pickle['policy_epochs'] = policy_epochs
            self.to_pickle['check_early_stop'] = check_early_stop
            self.to_pickle['value_replay_buffer_mult'] = value_replay_buffer_mult
            self.to_pickle['policy_replay_buffer_mult'] = policy_replay_buffer_mult
            self.to_pickle['inpvecdim'] = self.to_pickle['obs_dim'] + self.to_pickle['act_dim'] + 4 + 1 + 1 + self.to_pickle['act_dim'] + self.to_pickle['act_dim']
            self.to_pickle['normalize_obs'] = normalize_obs
            self.to_pickle['name'] = name
            self.to_pickle['best_reward'] = -np.infty
            self.to_pickle['value_batch_size'] = value_batch_size

            # Scaling was not used in the context of the paper's experiments. Scaling could have even further
            # enhanced the results
            self.to_pickle['scaler_vars'] = np.zeros(self.to_pickle['obs_dim'])
            self.to_pickle['scaler_means'] = np.zeros(self.to_pickle['obs_dim'])
            self.to_pickle['scaler_m'] = 0
            self.to_pickle['scaler_n'] = 0
            self.to_pickle['scaler_first_pass'] = True

            # All the information needed for a given time-step is stored in a vector
            # That is, for each time-step we store the state, action, reward, discounted sum reward, critic's output
            # .... in a vector. Then a trajectory (multiple time-steps) is defined as a list of such vectores.
            # Here we define the indices to access a particular information of a given time-step.
            self.to_pickle['indexes'] = {}
            self.to_pickle['indexes']['obs'] = [0, self.to_pickle['obs_dim']]
            self.to_pickle['indexes']['act'] = [self.to_pickle['indexes']['obs'][-1], self.to_pickle['indexes']['obs'][-1] + self.to_pickle['act_dim']]
            self.to_pickle['indexes']['rew'] = [self.to_pickle['indexes']['act'][-1], self.to_pickle['indexes']['act'][-1] + 1]
            self.to_pickle['indexes']['disc_sum_rew'] = [self.to_pickle['indexes']['rew'][-1], self.to_pickle['indexes']['rew'][-1] + 1]
            self.to_pickle['indexes']['value'] = [self.to_pickle['indexes']['disc_sum_rew'][-1], self.to_pickle['indexes']['disc_sum_rew'][-1] + 1]
            self.to_pickle['indexes']['advantages'] = [self.to_pickle['indexes']['value'][-1], self.to_pickle['indexes']['value'][-1] + 1]
            self.to_pickle['indexes']['beta'] = [self.to_pickle['indexes']['advantages'][-1], self.to_pickle['indexes']['advantages'][-1] + 1]
            self.to_pickle['indexes']['eta'] = [self.to_pickle['indexes']['beta'][-1], self.to_pickle['indexes']['beta'][-1] + 1]
            self.to_pickle['indexes']['old_means'] = [self.to_pickle['indexes']['eta'][-1], self.to_pickle['indexes']['eta'][-1] + self.to_pickle['act_dim']]
            self.to_pickle['indexes']['old_vars'] = [self.to_pickle['indexes']['old_means'][-1], self.to_pickle['indexes']['old_means'][-1] + self.to_pickle['act_dim']]

            self.to_pickle['classic_observations'] = [[-1, self.to_pickle['indexes']['obs'][0],
                                                       self.to_pickle['indexes']['obs'][0] + env.nofb_obs_dim]]
            self.to_pickle['fb_observations'] = [[-1, self.to_pickle['indexes']['obs'][0] + env.nofb_obs_dim,
                                                  self.to_pickle['indexes']['obs'][1]]]
            self.to_pickle['full_obs'] = [[-1, self.to_pickle['indexes']['obs'][0], self.to_pickle['indexes']['obs'][1]]]
            self.to_pickle['rewards_history'] = []
            self.to_pickle['RAM_usage'] = []
            self.to_pickle['compute_time'] = []
        # In case the test had previously been run, one can reload the pickle information and immediately get the
        # previous manager's state.
        else:
            self.load_pickle(suffix = suffix, name = name)
            self.load_networks(suffix = suffix, model_suffix = model_suffix, cpu_only= cpu_only)

    # In case networks had previously been created, load the corresponding file based on the name and suffix.
    # suffix can correspond to the initial model has previously built or to the latest checkpoint (as the
    # network was left after previous training)
    def init_networks(self, suffix = '', name = '', cpu_only = False):
        self.load_pickle(suffix = suffix, name = name)
        self.load_networks(suffix = suffix, cpu_only= cpu_only)

    # Networks has never been created, so we need to build them. Let's build the actor-critic's structure
    # and define all the functions that need to be associated.
    def build_networks(self, policy_types, value_types, policy_sizes, value_sizes,
                       policy_params, value_params, policy_linker, value_linker,
                       init_policy_logvariance = -1.0, init_eta = 50.0, init_beta = 1.0,
                       kl_target = 0.003, value_lr = 6e-2, policy_lr = 2e-4, init_plr_mult = 1e-2,
                       time_steps = 50, cpu_only = False, save_policy_every = 10):
        self.ppo_networks = PPO_networks(self.to_pickle['obs_dim'], self.to_pickle['act_dim'], self.to_pickle['indexes'], self.to_pickle['inpvecdim'],
                                         policy_types, value_types, policy_sizes, value_sizes,
                                         policy_params, value_params, policy_linker, value_linker,
                                         init_policy_logvariance = init_policy_logvariance, init_eta = init_eta,
                                         init_beta = init_beta, kl_target = kl_target, value_lr = value_lr,
                                         policy_lr = policy_lr, init_plr_mult = init_plr_mult, time_steps = time_steps,
                                         cpu_only = cpu_only, save_policy_every = save_policy_every, name = self.to_pickle['name'])
        # Save the now-initialised network as the initial model for further potential use.
        self.save_results('_initialmodel')

    # Number of parameters in the model
    def get_number_params(self):
        return self.ppo_networks.number_params()

    # Runs the policy on nbr_episodes in parallel (nbr_episodes environments need to have been instantiated)
    # and build the corresponding trajectories with the states, actions and rewards filled for each time-step.
    def run_parallel_policy(self, nbr_episodes, gamma = None, params = None):
        dones = [False for _ in range(nbr_episodes)]
        if gamma == None:
            gamma = 0
        scale, offset = self.get_scaler()
        policy = self.ppo_networks
        episode_vec = []
        unscaled_obs = []
        states = None
        obs = np.array([el.reset(params) for el in self.parallel_envs[:nbr_episodes]])
        eval_values = [[] for _ in range(nbr_episodes)]
        lengths = np.zeros(nbr_episodes)
        length = 0
        while not np.all(dones):
            current_steps = np.zeros((nbr_episodes, self.to_pickle['inpvecdim']))
            obs = obs.astype(np.float32)
            unscaled_obs.append(obs)
            if self.to_pickle['normalize_obs']:
                obs_scaled = (obs-offset)*scale
                current_steps[:, self.to_pickle['indexes']['obs'][0]:self.to_pickle['indexes']['obs'][-1]] = obs_scaled
            else:
                current_steps[:, self.to_pickle['indexes']['obs'][0]:self.to_pickle['indexes']['obs'][-1]] = obs
            actions, states, _ = policy.get_action(current_steps, state = states)
            current_steps[:,self.to_pickle['indexes']['act'][0]:self.to_pickle['indexes']['act'][-1]] = actions
            length += 1

            next_obs = []
            for cnt, env, action in zip(range(nbr_episodes), self.parallel_envs, actions):
                if not dones[cnt]:
                    new_obs, reward, done, eval_val = env.take_step(action)
                    reward = np.float32(reward)
                    current_steps[cnt,
                    self.to_pickle['indexes']['rew'][0]:self.to_pickle['indexes']['rew'][-1]] = reward
                    if done:
                        lengths[cnt] = length
                        eval_values[cnt] = eval_val
                        dones[cnt] = True
                else:
                    new_obs = np.zeros(self.to_pickle['obs_dim'])
                next_obs.append(new_obs)

            obs = np.array(next_obs)
            episode_vec.append(current_steps) # Shape : [max_env_steps, batch_size, obs_dim]
        trajectories = np.stack(episode_vec, axis = 1) # Shape : [Batch_size, max_env_steps, obs_dim]
        trajectories = [el[:l] for el,l in zip(trajectories, lengths.astype(np.int32))]
        if self.to_pickle['normalize_obs']:
            unscaled_obs = np.stack(unscaled_obs, axis = 1)
            unscaled_obs_traj = [el[:l] for el,l in zip(unscaled_obs, lengths)]
            self.update_scaler(np.concatenate(unscaled_obs_traj))
        for t, v in zip(trajectories, eval_values):
            r = np.sum([el[self.to_pickle['indexes']['rew'][0]:self.to_pickle['indexes']['rew'][1]] for el in t])
            dr = polyval(gamma, [el[self.to_pickle['indexes']['rew'][0]] for el in t])
            self.to_pickle['rewards_history'].append([r,dr,v])
        #print (self.to_pickle['rewards_history'])
        mean_reward = np.mean([np.sum([el[self.to_pickle['indexes']['rew'][0]] for el in t]) for t in trajectories])
        print('Mean reward of : ', mean_reward)
        print('Mean evaluation value of : ', np.mean(eval_values))
        return trajectories, mean_reward

    # Initialise as many environments as necessary such that all episodes for a parameter update can be
    # played using parallelisable calls to tensorflow.
    def init_parallel_environments(self, nbr_envs):
        self.parallel_envs = [copy.deepcopy(self.env) for _ in range(nbr_envs)]

    # Part of the non-parallel version of run_parallel policy,
    # but allows to gather more information through elements_info in case it
    # is needed (for example, to analyze activation functions' evolution).
    def run_episode(self, show = False, params = None, elements_info = [], fix_state_switch = []):
        infos = []
        policy = self.ppo_networks
        scale, offset = self.get_scaler()
        obs = self.env.reset(params)
        episode_vec = []
        unscaled_obs = []
        done = False
        state = None
        eval_value = 0.0
        step = 0
        update_states = True
        not_yet_fixed = True
        while not done:
            current_step = np.zeros(self.to_pickle['inpvecdim'])
            obs = obs.astype(np.float32).reshape((-1))
            unscaled_obs.append(obs)
            obs_scaled = (obs - offset) * scale # center and scale observations, not used in the context of the paper
            if self.to_pickle['normalize_obs']:
                current_step[self.to_pickle['indexes']['obs'][0]:self.to_pickle['indexes']['obs'][-1]] = obs_scaled
            else:
                current_step[self.to_pickle['indexes']['obs'][0]:self.to_pickle['indexes']['obs'][-1]] = obs

            # Allows to do some test and freeze the neuromodulation signal
            if step in fix_state_switch:
                update_states = not update_states
                if update_states:
                    print ("States are being updated ! ")
                else:
                    print ("States are now fixed ! (No nmd or fixed G_previous)")
                fb_obs = current_step[self.to_pickle['indexes']['obs'][0] + self.to_pickle['nofb_obs_dim']:
                                      self.to_pickle['indexes']['obs'][1]]
            if not update_states:
                #print ("Not updating states !")
                current_step[
                self.to_pickle['indexes']['obs'][0] + self.to_pickle['nofb_obs_dim']:self.to_pickle['indexes']['obs'][
                    1]] = fb_obs
                actions, _, info = policy.get_action([current_step], state=state, elements_info=elements_info)
            else:
                actions, state, info = policy.get_action([current_step], state=state, elements_info=elements_info)

            infos.append(info)
            actions = actions[0]
            actions = actions.reshape((-1))
            current_step[self.to_pickle['indexes']['act'][0]:self.to_pickle['indexes']['act'][-1]] = actions
            obs, reward, done, eval_value = self.env.take_step(actions)
            if show:
                self.env.render()
            if not isinstance(reward, float):
                reward = float(reward)
            current_step[self.to_pickle['indexes']['rew'][0]:self.to_pickle['indexes']['rew'][-1]] = np.float64(reward)
            episode_vec.append(current_step)
            step += 1
        r_traj = [el[self.to_pickle['indexes']['rew'][0]] for el in np.array(episode_vec)]
        r = np.sum(r_traj)
        self.to_pickle['rewards_history'].append([r, r_traj, eval_value])
        print (self.to_pickle['rewards_history'])
        return np.array(episode_vec), np.concatenate(unscaled_obs), eval_value, infos

    # This marged with the run_episode provides the full non-parallel version of  run_parallel policy
    def run_policy(self, nbr_episodes):
        trajectories = []
        unscaled_obs = []
        eval_values = []
        for e in range(nbr_episodes):
            episode_vec, uobs, evaluation_value, infos = self.run_episode()
            trajectories.append(episode_vec)
            unscaled_obs.append(uobs)
            eval_values.append(evaluation_value)
        self.update_scaler(np.concatenate(unscaled_obs))
        mean_reward = np.mean([np.sum([el[self.to_pickle['indexes']['rew'][0]] for el in t]) for t in trajectories])
        print ('Mean reward of : ', mean_reward)
        print ('Mean evaluation value of : ', np.mean(eval_values))
        return trajectories, mean_reward

    # Discounted sum reward
    def discount(self, x, gamma):
        return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

    # Adds the discounted sum reward to all time-steps of trajectories
    def add_disc_sum_rew(self, trajectories, gamma):
        for t in trajectories:
            if gamma < 0.999:
                rewards = t[:,self.to_pickle['indexes']['rew'][0]] * (1-gamma)
            else:
                rewards = t[:,self.to_pickle['indexes']['rew'][0]]
            disc_sum_rew = self.discount(rewards, gamma)
            t[:,self.to_pickle['indexes']['disc_sum_rew'][0]] = disc_sum_rew

    # Add the critic's rating to all time-steps of trajectories
    def add_value(self, trajectories):
        value_function = self.ppo_networks
        values = value_function.get_value(trajectories)
        for t, vt in zip(trajectories, values):
            t[:,self.to_pickle['indexes']['value'][0]:self.to_pickle['indexes']['value'][1]] = vt

    # Add generalised advantage estimates to all time-steps of trajectories
    def add_gae(self, trajectories, gamma, lam):
        for t in trajectories:
            if gamma < 0.999:
                rewards = t[:, self.to_pickle['indexes']['rew'][0]] * (1-gamma)
            else:
                rewards = t[:, self.to_pickle['indexes']['rew'][0]]
            values = t[:,self.to_pickle['indexes']['value'][0]]
            tds = rewards - values + np.append(values[1:]*gamma, 0)
            advantages = self.discount(tds, gamma*lam)
            t[:, self.to_pickle['indexes']['advantages'][0]] = advantages

    # Normalize the GAE values amongs all time-steps of trajectories
    def normalize_advantages(self, trajectories):
        flat_advantages = np.concatenate([t[:,self.to_pickle['indexes']['advantages'][0]] for t in trajectories])
        mean = np.mean(flat_advantages)
        stddev = np.std(flat_advantages)
        for t in trajectories:
            t[:,self.to_pickle['indexes']['advantages'][0]] = (t[:,self.to_pickle['indexes']['advantages'][0]] - mean) / (stddev + 1e-6)

    # Runs the meta-RL procedure !
    def train(self, num_episodes = 20000, batch_size = 50, gamma = 0.995, lam = 0.98, parallel = True):
        # Build actor and criticm odels
        policy = self.ppo_networks
        value_trajectories = []
        policy_trajectories = []
        final_episode = self.to_pickle['episode'] + num_episodes
        if parallel:
            self.init_parallel_environments(batch_size)
        if not parallel:
            _, _ = self.run_policy(20)
        else:
            _, _ = self.run_parallel_policy(20, gamma=gamma)
        cnt = 0
        with open('./ReinforcementLearning/SavedModels/PPOModels/' + self.to_pickle['name'] + '_logfile', "w") as f:
            while self.to_pickle['episode'] < final_episode:
                f.write('################ Episode ')
                f.write(str(self.to_pickle['episode']))
                f.write('##################\n')
                start_global = time.time()
                if not parallel:
                    trajectories, reward = self.run_policy(batch_size)
                else:
                    trajectories, reward = self.run_parallel_policy(batch_size, gamma=gamma)
                # Run policy and compute advantages
                self.add_value(trajectories)
                self.add_disc_sum_rew(trajectories, gamma)
                self.add_gae(trajectories, gamma, lam)

                # Implement L'
                if self.to_pickle['trajectory_train_cut'] > 0 and self.to_pickle['trajectory_train_cut'] < len(
                        trajectories[0]):
                    trajectories = [el[:self.to_pickle['trajectory_train_cut']] for el in trajectories]

                # Normalize the advantages over the L' first time-steps (the only used for gradient descent)
                self.normalize_advantages(trajectories)
                self.to_pickle['episode'] += batch_size

                # Update informations
                f.write('Running envs took : ')
                f.write(str( time.time()-start_global))
                f.write('\n')
                start = time.time()
                f.write('Updating Policy\n')

                # Implement the replay buffer for the critic. (policy_replay_buffer = 1)
                value_trajectories += trajectories
                policy_trajectories += trajectories
                value_trajectories = value_trajectories[-self.to_pickle['value_replay_buffer_mult']*batch_size:]
                policy_trajectories = policy_trajectories[-self.to_pickle['policy_replay_buffer_mult']*batch_size:]

                # Update policy's parameters
                policy.update_policy(policy_trajectories, nbr_epochs=self.to_pickle['policy_epochs'],
                                    batch_size=-1,
                                    check_early_stop=self.to_pickle['check_early_stop'])
                f.write('Update took : ')
                f.write(str(time.time()-start))
                f.write('\n')

                f.write('Updating value function\n')

                # Update critic's parameters
                start = time.time()
                policy.update_value_function(value_trajectories, nbr_epochs = self.to_pickle['value_epochs'], batch_size=self.to_pickle['value_batch_size'])

                # Save parameters every 5 updates
                if cnt % 25 == 0:
                    cnt = 0
                    self.save_only_updates('_checkpoint')
                cnt += 1
                f.write('Update took : ')
                f.write(str(time.time()-start))
                f.write('\n')
                f.write('Time taken : ')
                f.write(str(time.time()-start_global))
                f.write('\n')
                # Add information, just in case
                self.to_pickle['RAM_usage'].append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                self.to_pickle['compute_time'].append(time.time()-start_global)
                f.flush()
        # Save the final parameters
        self.save_only_updates('_finalmodel')

    # Save meta_graph as well as parameters
    def save_results(self, suffix = ''):
        self.ppo_networks.save_models('./ReinforcementLearning/SavedModels/PPOModels/' + self.to_pickle['name'] + suffix)
        pickle.dump(self.to_pickle,
                    open('./ReinforcementLearning/SavedModels/PPOModels/' + self.to_pickle['name'] + suffix, 'wb'))

    # Only save parameters
    def save_only_updates(self, suffix = ''):
        self.ppo_networks.save_params('./ReinforcementLearning/SavedModels/PPOModels/' + self.to_pickle['name'] + suffix)
        pickle.dump(self.to_pickle,
                    open('./ReinforcementLearning/SavedModels/PPOModels/' + self.to_pickle['name'] + suffix, 'wb'))

    # Loads meta_graph from a given file and then restores parameters from another file
    def load_networks(self, suffix = '', model_suffix = '_initialmodel', cpu_only = False):
        mfn = './ReinforcementLearning/SavedModels/PPOModels/' + self.to_pickle['name'] + model_suffix
        self.ppo_networks = PPO_networks(self.to_pickle['obs_dim'], self.to_pickle['act_dim'], self.to_pickle['indexes'], self.to_pickle['inpvecdim'],[], [], [], [], [], [], [], [], file_name=mfn, cpu_only = cpu_only)
        fn = './ReinforcementLearning/SavedModels/PPOModels/' + self.to_pickle['name'] + suffix
        self.ppo_networks.restore_params(fn)

    # Reloads the pickled hyper-parameters
    def load_pickle(self, suffix = '', name = ''):
        self.to_pickle = pickle.load(open('./ReinforcementLearning/SavedModels/PPOModels/' + name + suffix, 'rb'))

    # Runs an episode with pygame graphical display
    def show_sample(self):
        trajs = self.run_episode(show = True)

    # Runs an episode from which it is possible to gather information as specified in elements_info and
    # for which it is possible to pass params to the environment allowing for modular testing
    # after training.
    def run_sample(self, show = False, params = None, elements_info = [], fix_state_switch = []):
        trajs, unsacled_obs, measure_value, infos = self.run_episode(show = show, params = params,elements_info=elements_info,
                                                                     fix_state_switch = fix_state_switch)
        obs = trajs[:,self.to_pickle['indexes']['obs'][0]:self.to_pickle['indexes']['obs'][1]]
        act = trajs[:,self.to_pickle['indexes']['act'][0]:self.to_pickle['indexes']['act'][1]]
        rewards = trajs[:,self.to_pickle['indexes']['rew'][0]:self.to_pickle['indexes']['rew'][1]]
        return obs, act, rewards, infos

    # This was taken from https://github.com/pat-coady/trpo but is not used in the context of the paper.
    def update_scaler(self, x):
        if self.to_pickle['scaler_first_pass']:
            self.to_pickle['scaler_means'] = np.mean(x, axis=0)
            self.to_pickle['scaler_vars'] = np.var(x, axis=0)
            self.sclaer_m = x.shape[0]
            self.to_pickle['scaler_first_pass'] = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.to_pickle['scaler_means'] * self.to_pickle['scaler_m']) + (new_data_mean * n)) / (self.to_pickle['scaler_m'] + n)
            self.to_pickle['scaler_vars'] = (((self.to_pickle['scaler_m'] * (self.to_pickle['scaler_vars'] + np.square(self.to_pickle['scaler_means']))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.to_pickle['scaler_m'] + n) -
                         np.square(new_means))
            self.to_pickle['scaler_vars'] = np.maximum(0.0, self.to_pickle['scaler_vars'])  # occasionally goes negative, clip
            self.to_pickle['scaler_means'] = new_means
            self.to_pickle['scaler_m'] += n

    def get_scaler(self):
        return 1 / (np.sqrt(self.to_pickle['scaler_vars']) + 0.1) / 3, self.to_pickle['scaler_means']

    def close(self):
        self.ppo_networks.close()

    def set_env(self, env):
        self.env = env


