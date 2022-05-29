import numpy as np
from scipy.stats import beta
import random
import pandas as pd
import plotnine as gg
import matplotlib.pyplot as plt
import pylab

from env.graph import Graph
from agent.var_ids import VarIDSAgent
from agent.var_ids import var_ids_action

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
    agent,
    plot_results=False):
    
  T = 7
  env = Graph(max_length=T, rewarding_state=3)
  
  num_state = 2*T - 1
  horizon = T - 1
  num_action = 2
  sims = range(num_sims)
  episodes = range(num_episodes)
  num_timesteps = num_episodes*horizon
  actions = list(range(num_action))

  action_count = []
  
  for sim in sims:
    rng = np.random.default_rng(sim)
    
    agent.reset(rng, obs_masks)
    action_count += [[num_timesteps*[0] for a in actions]]
    for episode in episodes:
      a = agent.select_action() 
      # Generate observation for all arms.
      obs = 
      # Mask the arms that are not observed.
      mask = 
      obs[np.logical_not(mask)] = np.nan
      agent.update(a, obs)

      action_count[sim][a][timestep] = 1

  action_count1 = [[float(action_count[0][a][t]) 
    for t in range(num_timesteps)] for a in actions]
  action_count = [[sum([float(action_count[sim][a][t]) 
    for sim in sims]) / num_sims 
    for t in range(num_timesteps)] for a in actions]

  # plot action frequencies averaged over simulations
  for a in actions:
    plt.plot(timesteps,
             pd.Series(action_count[a]).rolling(10, min_periods=1).mean(),
             colors[a], label='$a =$' + str(a+1))
  plt.axis([0,num_timesteps+1,0.0,1.01])
  plt.xlabel(r'time $t$', fontsize=20)
  plt.ylabel('$\mathbb{P}(A_t = a|\mathcal{E})$', fontsize=20)
#  plt.ylabel('$\mathrm{E}[N_{t,a} - N_{t-1,a}|\mathcal{E}]$', fontsize=20)
  pylab.legend(loc='best')
  plt.show()

  # plot action counts over single simulation
  for a in actions:
    plt.plot(timesteps, np.cumsum(action_count1[a]), colors[a], label='$a =$' + str(a+1))
  plt.axis([0,num_timesteps+1,0.0,num_timesteps+1])
  plt.xlabel(r'time $t$', fontsize=20)
  plt.ylabel('$N_{t,a}$', fontsize=20)
  pylab.legend(loc='best')
  plt.show()

  # plot action counts averaged over simulations
  for a in actions:
    plt.plot(timesteps, np.cumsum(action_count[a]), colors[a], label='$a =$' + str(a+1))
  plt.axis([0,num_timesteps+1,0.0,num_timesteps+1])
  plt.xlabel(r'time $t$', fontsize=20)
  plt.ylabel('$\mathrm{E}[N_{t,a}|\mathcal{E}]$', fontsize=20)
  pylab.legend(loc='best')
  plt.show()

  return action_count

if __name__ == '__main__':
    # Variance-IDS agent that learns from reward.
    var_ids_agent = VarIDSAgent(compute_action=var_ids_action)
    num_sims = 1000 # number of simulations over which to average
    num_timesteps = 1000 # number of time steps to simulate

    action_count_ts_reward = run_graph_rl_experiment(
        num_sims, 
        num_timesteps, 
        agent=var_ids_agent,
        plot_results=True)