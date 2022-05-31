import numpy as np
from scipy.stats import beta
import random
import pandas as pd
import plotnine as gg
import matplotlib.pyplot as plt
import pylab
from statistics import mean

from env.graph import Graph
from agent.var_ids import VarIDSAgent
from agent.var_ids import ts_action, var_ids_pess_action

colors = {0: 'b',
          1: 'g',
          2: 'r',
          3: 'c',
          4: 'm',
          5: 'y',
          6: 'k'}

#@title Utilities for running the experiments
def run_graph_rl_experiment(
    num_sims: int, 
    num_episodes: int, 
    T: int,
    agent,
    plot_results=False):
    
  epsilon = 0.9
  rewarding_state = 2*T-4
  env = Graph(max_length=T+1, rewarding_state=rewarding_state)
  
  num_state = 2 * T + 1
  horizon = T + 1
  num_action = 2
  sims = range(num_sims)
  episodes = range(num_episodes)
  num_timesteps = num_episodes*horizon
  timesteps = range(num_timesteps)
  actions = list(range(num_action))

  action_count = []
  info_ratio_all = []
  min_episodes = []
  shortfall_all = []
  mutual_info_all = []

  for sim in sims:
    rng = np.random.default_rng( )
    regret = 0
    agent.reset(rng, num_state)
    action_count += [[[num_episodes*[np.nan] for a in actions] for s in range(num_state)]]
    info_ratio_all += [[num_episodes*[np.full((2,2), np.nan)] for s in range(num_state)]]
    shortfall_all += [[num_episodes*[np.full((2,2), np.nan)] for s in range(num_state)]]
    mutual_info_all += [[num_episodes*[np.full((2,2), np.nan)] for s in range(num_state)]]
    for epi in episodes:
      if epi != 0:
        assert env.done
      env.reset()
      for hh in range(horizon):
        prev_state = agent.state
        a, action_stats = agent.select_action(hh) 
        # Generate observation for all arms.
        next_state, obs, bad_obs, done = env.step(a)
        regret -= agent.update(a, next_state, obs, bad_obs)
        action_count[sim][prev_state][a][epi] = 1
        info_ratio_all[sim][prev_state][epi] = action_stats['info_ratio']
        shortfall_all[sim][prev_state][epi] = action_stats['shortfall']
        mutual_info_all[sim][prev_state][epi] = action_stats['mutual_info']

        # if epi*horizon+hh % 100==0:
        #   alpha = action_stats['alpha']
        #   shortfall = action_stats['shortfall']
        #   mi = action_stats['mutual_info']
        #   print(f' State: {prev_state}, Action: {a}, Alpha: {alpha}, \n Shortfall: {shortfall}, \n MI: {mi}')

        if prev_state == rewarding_state and  epi % 10==0:
          alpha = action_stats['alpha']
          shortfall = action_stats['shortfall']
          mi = action_stats['mutual_info']
          Q_hat = action_stats['Q_hat'][:,prev_state,:].mean(0)
          V_hat = action_stats['V_hat'][:,prev_state].mean(0)
          Q_cross = action_stats['Q_cross'][:,prev_state,a].mean(0)
          Q_cross_var = action_stats['Q_cross'][:,prev_state,:].var(0)
          print(f'Episode: {epi}, State: {prev_state}, Action: {a}, Alpha: {alpha}, \n Shortfall: {shortfall}, \n MI: {mi}, \n Q_hat: {Q_hat}, \n V_hat: {V_hat}, \n Q_cross: {Q_cross}, \n Q_cross variance: {Q_cross_var}')
          # print(f' State: {prev_state}, Action: {a}, \n Shortfall: {shortfall}, \n Q_hat: {Q_hat}, \n V_hat: {V_hat}, \n Q_cross: {Q_cross}, \n Q_cross variance: {Q_cross_var}')


      regret += 1
      if ((regret / (epi+1)) < epsilon):
        min_episodes.append(epi+1)
        print(f'Sim: {sim}, Average Regret: {(regret / (epi+1))}')
        break

  action_count1 = [[[(action_count[0][s][a][t]) 
    for t in range(num_episodes)] for a in actions] for s in range(num_state)]
  action_count = [[[np.nanmean([(action_count[sim][s][a][t]) 
    for sim in sims], axis=0)
    for t in range(num_episodes)] for a in actions] for s in range(num_state)]
  info_ratio_all = [[np.nanmean([(info_ratio_all[sim][s][t]) 
    for sim in sims], axis=0)
    for t in range(num_episodes)] for s in range(num_state)]
  shortfall_all = [[np.nanmean([(shortfall_all[sim][s][t]) 
    for sim in sims], axis=0)
    for t in range(num_episodes)] for s in range(num_state)]
  mutual_info_all = [[np.nanmean([(mutual_info_all[sim][s][t]) 
    for sim in sims], axis=0)
    for t in range(num_episodes)] for s in range(num_state)]
  # assert len(min_episodes) == num_sims, min_episodes
  min_episode = mean(min_episodes)
  print(f'Number of episodes until average regret < {epsilon} = {min_episode}')
  
  # # plot action frequencies averaged over simulations
  # for s in range(num_state):
  #   for a in actions:
  #     plt.plot(episodes,
  #             pd.Series(action_count[s][a]).rolling(10, min_periods=1).mean(),
  #             colors[a], label='$a =$' + str(a+1))
  #   plt.axis([0,num_timesteps+1,0.0,1.01])
  #   plt.xlabel(r'time $t$', fontsize=20)
  #   plt.ylabel('$\mathbb{P}(A_t = a|\mathcal{E})$', fontsize=20)
  # #  plt.ylabel('$\mathrm{E}[N_{t,a} - N_{t-1,a}|\mathcal{E}]$', fontsize=20)
  #   pylab.legend(loc='best')
  #   plt.show()

  # plot action counts over single simulation
  fig, axes = plt.subplots(horizon, 1, figsize=(10, 5*horizon))
  for s in range(horizon):
    for a in actions:
      axes[s].plot(episodes, np.cumsum(action_count1[2*s][a]), colors[a], label='$a =$' + str(a+1))
    axes[s].set_title(f'State {2*s} action selection', fontsize=20)
    axes[s].set_xlabel('Time $t$', fontsize=20)
    axes[s].set_ylabel('$N_{t,s,a}$', fontsize=20)
  plt.axis([0,num_timesteps+1,0.0,num_timesteps+1])
  pylab.legend(loc='best')
  plt.tight_layout()
  # plt.show()
  plt.savefig('action_count_single')

  # plot action counts averaged over simulations
  fig, axes = plt.subplots(horizon, 1, figsize=(10, 5*horizon))
  for s in range(horizon):
    for a in actions:
      axes[s].plot(episodes, np.cumsum(action_count[2*s][a]), colors[a], label='$a =$' + str(a+1))
    axes[s].set_title(f'State {2*s} action selection', fontsize=20)
    axes[s].set_xlabel('Time $t$', fontsize=20)
    axes[s].set_ylabel('$\mathrm{E}[N_{t,s,a}|\mathcal{E}]$', fontsize=20)
  plt.axis([0,num_timesteps+1,0.0,num_timesteps+1])
  pylab.legend(loc='best')
  plt.tight_layout()
  # plt.show()
  plt.savefig('action_count_expectation')


  fig, axes = plt.subplots(horizon, 1, figsize=(10, 5*horizon))
  for s in range(horizon):
    for a1 in range(num_action):
      for a2 in range(num_action):
        axes[s].plot(episodes, [info_ratio_all[2*s][t][a1,a2] for t in episodes], colors[a1+2*a2], label=f'$a1 =${a1}, $a2 =${a2}')
        axes[s].set_title(f'State {2*s} Action ({a1},{a2}) info ratio', fontsize=20)
        axes[s].set_xlabel('Episode $t$', fontsize=20)
        axes[s].set_ylabel('Expected Info Ratio', fontsize=20)
  plt.axis([0,num_episodes+1,0.0,num_episodes+1])
  pylab.legend(loc='best')
  plt.tight_layout()
  # plt.show()
  plt.savefig('info_ratio_expectation')

  fig, axes = plt.subplots(horizon, 1, figsize=(10, 5*horizon))
  for s in range(horizon):
    for a1 in range(num_action):
      for a2 in range(num_action):
        axes[s].plot(episodes, [shortfall_all[2*s][t][a1,a2] for t in episodes], colors[a1+2*a2], label=f'$a1 =${a1}, $a2 =${a2}')
        axes[s].set_title(f'State {2*s} Action ({a1},{a2}) shortfall', fontsize=20)
        axes[s].set_xlabel('Episode $t$', fontsize=20)
        axes[s].set_ylabel('Expected Shortfall', fontsize=20)
  plt.axis([0,num_episodes+1,0.0,num_episodes+1])
  pylab.legend(loc='best')
  plt.tight_layout()
  # plt.show()
  plt.savefig('shortfall_expectation')

  fig, axes = plt.subplots(horizon, 1, figsize=(10, 5*horizon))
  for s in range(horizon):
    for a1 in range(num_action):
      for a2 in range(num_action):
        axes[s].plot(episodes, [mutual_info_all[2*s][t][a1,a2] for t in episodes], colors[a1+2*a2], label=f'$a1 =${a1}, $a2 =${a2}')
        axes[s].set_title(f'State {2*s} Action ({a1},{a2}) mutual info', fontsize=20)
        axes[s].set_xlabel('Episode $t$', fontsize=20)
        axes[s].set_ylabel('Expected MI', fontsize=20)
  plt.axis([0,num_episodes+1,0.0,num_episodes+1])
  pylab.legend(loc='best')
  plt.tight_layout()
  # plt.show()
  plt.savefig('mutual_info_expectation')

  return action_count

if __name__ == '__main__':
    # Variance-IDS agent that learns from observation.
    var_ids_agent = VarIDSAgent(compute_action=var_ids_pess_action, pess_factor=0)
    num_sims = 100 # number of simulations over which to average
    num_episodes = 100 # number of time steps to simulate
    T = 10 # number of possible rewarding_states

    action_count_ts_reward = run_graph_rl_experiment(
        num_sims, 
        num_episodes,
        T,
        agent=var_ids_agent,
        plot_results=True)