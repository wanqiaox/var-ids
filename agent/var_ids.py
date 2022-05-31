import numpy as np
import abc

#@title Agent interface
class Agent(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def reset(self, rng: np.random.Generator, obs_masks: np.ndarray) -> None:
    # Reset the agent.
    raise NotImplementedError

  @abc.abstractmethod
  def update(self, action: int, obs: np.ndarray) -> None:
    # Update agent state.
    raise NotImplementedError

  @abc.abstractmethod
  def select_action(self) -> int:
    # Select an action.
    raise NotImplementedError

def random_argmin(rng, scores: np.ndarray):
  scores = scores.reshape(-1)
  probs = (scores==scores.min()).astype(np.float32)
  probs /= probs.sum()
  return rng.choice(np.arange(scores.size), p=probs)

def random_argmax(rng, scores: np.ndarray):
  scores = scores.reshape(-1)
  probs = (scores==scores.max()).astype(np.float32)
  probs /= probs.sum()
  return rng.choice(np.arange(scores.size), p=probs)

def value_iter(t, p, r):
  # if t == 4:
  #   print("ahh")
  num_action = p.shape[2]
  num_state = p.shape[1]
  num_samples = p.shape[0]
  H = (num_state + 1) // 2 # TODO: check
  V = np.zeros((num_samples, num_state,))
  for _ in range (H - 1, t-1, -1):
    Q = r + np.sum(p * V[:,None,None,:], axis=-1)
    V = np.max(Q, axis=-1)
#     Q = (p*(r + V[:,None]).transpose()[None]).sum(axis=-1) #shape: (state, actionn)
#     V = Q.max(axis=-1) #(state,)
  return Q, V

def var_ids_pess_action(rng, pess_factor, t, state, num_trans: np.ndarray, num_success: np.ndarray, num_failure: np.ndarray,
                   num_good_obs: np.ndarray, num_bad_obs: np.ndarray, num_sample: int = 32):
  num_action = num_trans.shape[1]
  num_state = num_trans.shape[0]
#   num_action = len(num_success)

#   p_samples = rng.beta(np.tile(num_success + 1, (num_sample, 1)), np.tile(num_failure + 1, (num_sample, 1)))
  trans_shape = num_trans.shape
  reward_shape = num_success.shape
  pp = np.zeros((num_state, num_action, num_state))
  for i in range(num_state-2):
    if i%2 ==0:
      pp[i,0,i+2] = 1
      pp[i,1,i+1] = 1
    else:
      pp[i,0,i+2]=1
      pp[i,1,i+2]=1
  # pp[num_state-2,0,num_state-2]=pp[num_state-1,0,num_state-1]=pp[num_state-2,1,num_state-2]=pp[num_state-1,1,num_state-1]=1
  pp[num_state-2,0,0]=pp[num_state-1,0,0]=pp[num_state-2,1,0]=pp[num_state-1,1,0]=1
  p_samples = np.tile(pp, (num_sample, 1,1,1))
  # p_samples = np.stack([np.stack([rng.dirichlet(num_trans[s,a], size=num_sample) for a in range(num_action)], axis=1) for s in range(num_state)], axis=1)
  r_samples = rng.beta(np.tile((num_success + 1).reshape(-1), (num_sample, 1)), np.tile((num_failure + 1).reshape(-1), (num_sample, 1)))
  r_samples = r_samples.reshape(num_sample, *reward_shape)
  # r_samples[:, np.arange(0, num_state, 2), 0] = 0
  # r_samples[:, np.arange(1, num_state, 2), :] = 0
  # r_samples[:,num_state-1,:] = 0
   
  c_samples = rng.beta(np.tile((num_good_obs + 1).reshape(-1), (num_sample, 1)), np.tile((num_bad_obs + 1).reshape(-1), (num_sample, 1)))
  c_samples = c_samples.reshape(num_sample, *reward_shape)
  # c_samples[:, np.arange(0, num_state, 2), 0] = 0
  # c_samples[:, np.arange(1, num_state, 2), :] = 0
  # c_samples[:,num_state-1,:] = 0

  Q_hat, V_hat = value_iter(t, p_samples.reshape(num_sample, *trans_shape), r_samples)
  Q_cross, _ = value_iter(t, p_samples.reshape(num_sample, *trans_shape), c_samples)

  # Minimize information ratio over 2-sparse policies
  A1 = (V_hat[:,state, None] - Q_hat[:,state,:]).mean(0)
  A2 = A1
  C1 = np.array([np.cov(Q_cross[:,state,a]) for a in range(num_action)])
  C2 = C1
  
  alpha_1 = (-np.sqrt((C1[None,:] - C2[:,None])**2*pess_factor + (A1[None,:]*C2[:,None] - A2[:,None]*C1[None,:])**2) +
   (A2[:,None] - A1[None,:])*C2[:,None])/((A1[None,:]-A2[:,None])*(C1[None,:] - C2[:,None])+  1e-15)
  # alpha_2 = (np.sqrt((C1[None,:] - C2[:,None])**2*pess_factor + (A1[None,:]*C2[:,None] - A2[:,None]*C1[None,:])**2) +
  #  (A2[:,None] - A1[None,:])*C2[:,None])/((A1[None,:]-A2[:,None])*(C1[None,:] - C2[:,None])+ 1e-15)
  # assert np.isclose(alpha_1[0][1] + alpha_2[1][0], 1)
  # assert np.isclose(alpha_1[1][0] + alpha_2[0][1], 1)
  alphas = np.ma.masked_where(~((0<=alpha_1)&(alpha_1<=1)), alpha_1)
  alphas[1,1] = 0
  alphas[0,0] = 1
  # alphas = np.where((0<=alphas)&(alphas<=1), alphas, 0)
  
  assert ((0<=alphas)&(alphas<=1)).all()
  f_alpha = (( A2[:,None] + (A1[None,:] - A2[:,None])*alphas)**2 + pess_factor)/(C2[:,None] + (C1[None,:] - C2[:,None])*alphas + 1e-15)
  f_0 = (A2[1]**2 + pess_factor)/ (C2[1]+1e-15)
  f_1 = (A1[0]**2 + pess_factor)/ (C1[0]+1e-15)
  f_alpha[0,0] = f_1
  f_alpha[1,1] = f_0

  # f_alpha = np.ma.masked_invalid(f_alpha)
  assert (f_alpha >= 0).all()
  # f_alpha[np.eye(num_action, dtype=bool)] = np.inf
  # f_0[np.eye(num_action, dtype=bool)] = np.inf
  # f_1[np.eye(num_action, dtype=bool)] = np.inf
  # f_matrix = np.stack([f_0, f_alpha, f_1], axis=0)
  index = random_argmax(rng, f_alpha)
  index = np.unravel_index(index, f_alpha.shape)
  # assert f_alpha[index[0],index[1]] == f_alpha.min()
    
  alpha = alphas[index[0], index[1]]
  assert not np.isnan(alpha)
  action = np.zeros(num_action)
  action[0] = alpha
  action[1] = 1 - alpha
  assert ((0<=action)&(action<=1)).all() and action.sum() == 1
  shortfall = np.zeros([num_action, num_action])
  shortfall[0,0] = A1[0]**2 + pess_factor
  shortfall[1,1] = A1[1]**2 + pess_factor
  shortfall[0,1] = (A1[0]*alpha + A1[1]*(1-alpha))**2 + pess_factor
  shortfall[1,0] = (A1[0]*(1-alpha) + A1[1]*alpha)**2 + pess_factor

  mutual_info = np.zeros([num_action, num_action])
  mutual_info[0,0] = C1[0]
  mutual_info[1,1] = C1[1]
  mutual_info[0,1] = (C1[0]*alpha + C1[1]*(1-alpha))
  mutual_info[1,0] = (C1[0]*(1-alpha) + C1[1]*alpha)


  return rng.choice(np.arange(num_action), p=action), {'info_ratio': f_alpha, 'shortfall': shortfall, 'mutual_info': mutual_info, 'alpha':alpha, 'Q_hat': Q_hat, 'V_hat': V_hat, 'Q_cross': Q_cross}

def ts_action(rng, pess_factor, t, state, num_trans: np.ndarray, num_success: np.ndarray, num_failure: np.ndarray,
                   num_good_obs: np.ndarray, num_bad_obs: np.ndarray, num_sample: int = 32):
  num_action = num_trans.shape[1]
  num_state = num_trans.shape[0]
#   num_action = len(num_success)

#   p_samples = rng.beta(np.tile(num_success + 1, (num_sample, 1)), np.tile(num_failure + 1, (num_sample, 1)))
  trans_shape = num_trans.shape
  reward_shape = num_success.shape
  pp = np.zeros((num_state, num_action, num_state))
  for i in range(num_state-2):
    if i%2 ==0:
      pp[i,0,i+2] = 1
      pp[i,1,i+1] = 1
    else:
      pp[i,0,i+2]=1
      pp[i,1,i+2]=1
  pp[num_state-2,0,num_state-2]=pp[num_state-1,0,num_state-1]=pp[num_state-2,1,num_state-2]=pp[num_state-1,1,num_state-1]=1
  p_samples = np.tile(pp, (num_sample, 1,1,1))
  # p_samples = np.stack([np.stack([rng.dirichlet(num_trans[s,a], size=num_sample) for a in range(num_action)], axis=1) for s in range(num_state)], axis=1)
  r_samples = rng.beta(np.tile((num_success + 1).reshape(-1), (num_sample, 1)), np.tile((num_failure + 1).reshape(-1), (num_sample, 1)))
  r_samples = r_samples.reshape(num_sample, *reward_shape)
  # r_samples[:, np.arange(0, num_state, 2), 0] = 0
  # r_samples[:, np.arange(1, num_state, 2), :] = 0
  # r_samples[:,num_state-1,:] = 0
   
  c_samples = rng.beta(np.tile((num_good_obs + 1).reshape(-1), (num_sample, 1)), np.tile((num_bad_obs + 1).reshape(-1), (num_sample, 1)))
  c_samples = c_samples.reshape(num_sample, *reward_shape)
  c_samples[:, np.arange(0, num_state, 2), 0] = 0
  c_samples[:, np.arange(1, num_state, 2), :] = 0
  c_samples[:,num_state-1,:] = 0

  Q_hat, V_hat = value_iter(t, p_samples.reshape(num_sample, *trans_shape), r_samples)
  Q_cross, _ = value_iter(t, p_samples.reshape(num_sample, *trans_shape), c_samples)

  # Minimize regret
  A1 = (V_hat[:,state, None] - Q_hat[:,state,:]).mean(0)
  
  action = random_argmin(rng, A1**2)
  assert A1[action] <= A1[1-action]

  return action, {'shortfall': np.stack([A1**2, A1**2], axis=-1), 'Q_hat': Q_hat, 'V_hat': V_hat, 'Q_cross': Q_cross}

class VarIDSAgent(Agent):
  def __init__(self, compute_action, pess_factor=0):
    ''' pess_factor controls how pessimistic you are '''
    self._compute_action = compute_action
    self._state = 0
    self._rng = None
    self._pess_factor = pess_factor
  @property
  def state(self):
    return self._state
  def reset(self, rng: np.random.Generator, num_state: int):
    self._num_action = 2
    self._num_state = num_state
    self._rng = rng
    self._state = 0
    
    self._num_trans = np.ones(shape=(self._num_state, self._num_action, self._num_state,)) / self._num_state
    self._num_success = np.zeros(shape=(self._num_state, self._num_action,))
    self._num_failure = np.zeros(shape=(self._num_state, self._num_action,))
    self._num_good_obs = np.zeros(shape=(self._num_state, self._num_action,))
    self._num_bad_obs = np.zeros(shape=(self._num_state, self._num_action,))
  
  def update(self, action: int, next_state, obs: np.ndarray, bad_obs:np.ndarray):
    self._num_trans[self.state, action, next_state] += 1
    reward = obs[self.state, action]
    self._num_success[self.state, action] += obs[self.state, action]
    self._num_failure[self.state, action] += 1 - obs[self.state, action]
    self._num_good_obs += np.nan_to_num(obs)
    self._num_bad_obs += np.nan_to_num(bad_obs)
    self._state = next_state
    return reward

  def select_action(self, t: int):
    
    return self._compute_action(self._rng, self._pess_factor, t, self._state, self._num_trans, self._num_success, self._num_failure,
                                self._num_good_obs, self._num_bad_obs)
    