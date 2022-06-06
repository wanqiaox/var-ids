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
from agent.var_ids import *

import dask
colors = {0: 'b',
          1: 'g',
          2: 'r',
          3: 'c',
          4: 'm',
          5: 'y',
          6: 'k'}
markers = {0: 'x',
          1: '^',
          2: 'o',
          3: 's',
          4: 'P',
          5: '*',
          6: '+'}
#@title Utilities for running the experiments
def run_graph_rl_experiment(
    num_sims: int, 
    num_episodes: int, 
    H: int,
    debug=False,
    plot_results=False):
    
  pess_factor = 0 # used as exp_factor(=1) when using exp action, beta(=1) when using softplus action
  epsilon = 0.9
  rewarding_state = H//2*2
  action_type = var_ids_pess_action # options: var_ids_pess_action, var_ids_exp, var_ids_softplus
  print(f'Horizon: {H}, Rewarding state: {rewarding_state}, Action type: {str(action_type)}, Pess factor: {pess_factor}')
  num_state = 2 * H - 1
  horizon = H
  num_action = 2
  sims = range(num_sims)
  episodes = range(num_episodes)
  num_timesteps = num_episodes*horizon
  timesteps = range(num_timesteps)
  actions = list(range(num_action))

  # action_count = [None for _ in range(num_sims)]
  # info_ratio_all = [None for _ in range(num_sims)]
  # min_episodes = [None for _ in range(num_sims)]
  # shortfall_all = [None for _ in range(num_sims)]
  # mutual_info_all = [None for _ in range(num_sims)]


  def simulate(sim, ):
    env = Graph(max_length=H, rewarding_state=rewarding_state)
    # agent = VarIDSAgent(compute_action=var_ids_pess_action, pess_factor=pess_factor)
    agent = VarIDSAgent(compute_action=action_type, pess_factor=pess_factor)

    rng = np.random.default_rng(sim*42 )
    regret = 0
    agent.reset(rng, num_state)
    action_count_per_sim = [[num_episodes*[0] for a in actions] for s in range(num_state)]
    info_ratio_all_per_sim = [num_episodes*[np.full((2,2), np.nan)] for s in range(num_state)]
    shortfall_all_per_sim =[num_episodes*[np.full((2,2), np.nan)] for s in range(num_state)]
    mutual_info_all_per_sim =[num_episodes*[np.full((2,2), np.nan)] for s in range(num_state)]
    for epi in range(num_episodes):
      if epi != 0:
        assert env.done
      env.reset()
      for hh in range(horizon):
        prev_state = agent.state
        a, action_stats = agent.select_action(hh) 
        # Generate observation for all arms.
        next_state, num_success, num_failure, obs, bad_obs, done = env.step(a)
        r = num_success[prev_state, a]
        regret -= r
        agent.update(a,  num_success, num_failure, next_state, obs, bad_obs)
        action_count_per_sim[prev_state][a][epi] = 1
        info_ratio_all_per_sim[prev_state][epi] = action_stats['info_ratio']
        shortfall_all_per_sim[prev_state][epi] = action_stats['shortfall']
        mutual_info_all_per_sim[prev_state][epi] = action_stats['mutual_info']

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
      
      agent.update_posterior()

      regret += 1
      if ((regret / (epi+1)) < epsilon):
        min_epi_to_reach_target = epi+1
        print(f'Early break! Sim: {sim}, Episode: {epi}, Average Regret: {(regret / ((epi+1)))}')
        break
    if epi+1 == num_episodes:
      min_epi_to_reach_target = epi+1
      print(f'Ran all episodes. Sim: {sim}, Episode: {epi}, Average Regret: {(regret / (epi+1))}, Regret at Cutoff: {regret}')
    _,rewarding_state_action_stats = agent._compute_action(agent._rng, pess_factor, H//2, 
    rewarding_state, agent._p_samples, agent._r_samples, agent._c_samples)
    shortfal_pess_rs = rewarding_state_action_stats['shortfall']
    
    _,info_state_action_stats = agent._compute_action(agent._rng, pess_factor, H-2, 
    2*H-4, agent._p_samples, agent._r_samples, agent._c_samples)
    shortfal_pess_is = info_state_action_stats['shortfall']
    final_shortfall_pess_ratio = (shortfal_pess_rs, shortfal_pess_is)
    return action_count_per_sim, info_ratio_all_per_sim, shortfall_all_per_sim, mutual_info_all_per_sim, min_epi_to_reach_target, final_shortfall_pess_ratio

  results = []
  for sim in range(num_sims):
    if debug:
      result = simulate(sim)
    else:
      result = dask.delayed(simulate)(sim)
    results.append(result)
  if not debug:
    results = dask.compute(*results)
  results = list(zip(*results))
  action_count, info_ratio_all, shortfall_all, mutual_info_all, min_episodes, final_shortfall_pess_ratio = results

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
  assert len(min_episodes) == num_sims, min_episodes
  min_episode = mean(min_episodes)
  shortfall_rewarding_mean = np.nanmean(shortfall_all[rewarding_state], axis=0)
  shortfall_informative_mean = np.nanmean(shortfall_all[2*H-4], axis=0)
  print(f'Number of episodes until average regret < {epsilon} = {min_episode}')
  print(f'Average shortfall/pess at rewarding state {rewarding_state} = {shortfall_rewarding_mean/pess_factor}')
  print(f'Average shortfall/pess at informative state {2*H-4} = {shortfall_informative_mean/pess_factor}')
  final_shortfall_rs_pess_ratio_mean = np.mean(final_shortfall_pess_ratio[0], axis=0)
  final_shortfall_is_pess_ratio_mean = np.mean(final_shortfall_pess_ratio[1], axis=0)
  print(f'Final shortfall/pess at rewarding state {rewarding_state} = {final_shortfall_rs_pess_ratio_mean/pess_factor}')
  print(f'Final shortfall/pess at informative state {2*H-4} = {final_shortfall_is_pess_ratio_mean/pess_factor}')
  # plot action counts over single simulation
  fig, axes = plt.subplots(horizon, 1, figsize=(10, 5*horizon))
  for s in range(horizon):
    for a in actions:
      axes[s].plot(episodes, np.cumsum(action_count1[2*s][a]), colors[a],  label='$a =$' + str(a+1))
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
    for a1,a2 in [(0,0),(1,1)]:
      axes[s].plot(episodes, [info_ratio_all[2*s][t][a1,a2] for t in episodes], colors[a1+2*a2], label=f'$a1 =${a1}, $a2 =${a2}')
    axes[s].set_title(f'State {2*s} info ratio', fontsize=20)
    axes[s].set_xlabel('Episode $t$', fontsize=20)
    axes[s].set_ylabel('Expected Info Ratio', fontsize=20)
    axes[s].legend(loc='best')
  plt.axis([0,num_episodes+1,0.0,num_episodes+1])
  pylab.legend(loc='best')
  plt.tight_layout()
  # plt.show()
  plt.savefig('info_ratio_expectation')

  fig, axes = plt.subplots(horizon, 1, figsize=(10, 5*horizon))
  for s in range(horizon):
    for a1,a2 in [(0,0),(1,1)]:
      axes[s].plot(episodes, [shortfall_all[2*s][t][a1,a2] for t in episodes], colors[a1+2*a2], label=f'$a1 =${a1}, $a2 =${a2}')
    axes[s].set_title(f'State {2*s} shortfall', fontsize=20)
    axes[s].set_xlabel('Episode $t$', fontsize=20)
    axes[s].set_ylabel('Expected Shortfall', fontsize=20)
    axes[s].legend(loc='best')
  plt.axis([0,num_episodes+1,0.0,num_episodes+1])
  pylab.legend(loc='best')
  plt.tight_layout()
  # plt.show()
  plt.savefig('shortfall_expectation')

  fig, axes = plt.subplots(horizon, 1, figsize=(10, 5*horizon))
  for s in range(horizon):
    for a1,a2 in [(0,0),(1,1)]:
      axes[s].plot(episodes, [mutual_info_all[2*s][t][a1,a2] for t in episodes], colors[a1+2*a2], label=f'$a1 =${a1}, $a2 =${a2}')
    axes[s].set_title(f'State {2*s} mutual info', fontsize=20)
    axes[s].set_xlabel('Episode $t$', fontsize=20)
    axes[s].set_ylabel('Expected MI', fontsize=20)
    axes[s].legend(loc='best')
  plt.axis([0,num_episodes+1,0.0,num_episodes+1])
  pylab.legend(loc='best')
  plt.tight_layout()
  # plt.show()
  plt.savefig('mutual_info_expectation')

  return action_count

if __name__ == '__main__':
    # Variance-IDS agent that learns from observation.
    num_sims = 10 # number of simulations over which to average
    num_episodes = 300 # number of time steps to simulate
    # T = 100 # number of possible rewarding_states
    debug = True
    for T in [60, 80, 100]:
      action_count_ts_reward = run_graph_rl_experiment(
          num_sims, 
          num_episodes,
          T+1,
          debug=debug,
          plot_results=True)