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

def value_iter(t, p, r):
  num_action = p.shape[2]
  num_state = p.shape[1]
  num_samples = p.shape[0]
  H = (num_state + 1) / 2 # TODO: check
  V = np.zeros((num_samples, num_state,))
  for _ in range (H - 1, t-1, -1):
    Q = r + np.sum(p * V, axis=-1)
    V = np.amax(Q, axis=-1)
#     Q = (p*(r + V[:,None]).transpose()[None]).sum(axis=-1) #shape: (state, actionn)
#     V = Q.max(axis=-1) #(state,)
  return Q, V

def var_ids_action(rng, t, state, num_trans: np.ndarray, num_success: np.ndarray, num_failure: np.ndarray,
                   num_good_obs: np.ndarray, num_bad_obs: np.ndarray, obs_masks: np.ndarray, num_sample: int = 32):
  num_action = num_trans.shape[1]
  num_state = num_trans.shape[0]
#   num_action = len(num_success)

#   p_samples = rng.beta(np.tile(num_success + 1, (num_sample, 1)), np.tile(num_failure + 1, (num_sample, 1)))
  trans_shape = num_trans.shape
  reward_shape = num_success.shape
  p_samples = rng.dirichlet(np.tile(num_trans.reshape(-1), (num_sample, 1)))
  r_samples = rng.beta(np.tile((num_success + 1).reshape(-1), (num_sample, 1)), np.tile((num_failure + 1).reshape(-1), (num_sample, 1)))
  c_samples = rng.beta(np.tile((num_good_obs + 1).reshape(-1), (num_sample, 1)), np.tile((num_bad_obs + 1).reshape(-1), (num_sample, 1)))
  
  Q_hat, V_hat = value_iter(t, p_samples.reshape(num_sample, *trans_shape), r_samples.reshape(num_samples, *reward_shape))
  Q_cross, _ = value_iter(t, p_samples.reshape(num_sample, *trans_shape), c_samples.reshape(num_samples, *reward_shape))

  # Minimize information ratio over 2-sparse policies
  info_ratio = np.zeros([num_action, num_action])
  
  A1 = (V_hat[:,state, None] - Q_hat[:,state,:]).mean(0)
  A2 = A1
  C1 = np.array([np.trace(np.cov(Q_cross[:,state,a])) for a in range(num_action)])
  C2 = C1

  f_alpha = 4*(A1[None,:] - A2[:,None])*(A2[:,None]*C1[None,:] - A1[None,:]*C2[:,None]) / (C1[None,:] - C2[:,None])**2
  f_0 = np.tile(A2[:,None]**2 / C2[:,None], (1, num_action))
  f_1 = np.tile(A1[None,:]**2 / C1[None,:], (num_action, 1))
  alphas = (A2[:,None]*C1[None,:] - 2*A1[None,:]*C2[:,None] + np.tile(A2[:,None]*C2[:,None], (1, num_action))) / (C1[None,:] - C2[:,None])*(A1[None,:] - A2[:,None])
  f_matrix = np.stack([f_0, f_alpha, f_1], axis=0)
  index = np.argmax(f_matrix)
  index = np.unravel_index(index, f_matrix.shape)
  assert f_matrix[index[0],index[1],index[2]] == f_matrix.max()
    
  alpha = alphas[index[1], index[2]] if index[0] == 1 else index[0]
  action = np.zeros(num_action)
  action[index[1]] = alpha
  action[index[2]] = 1 - alpha
  
  return rng.choice(np.arange(num_action), p=action)

class VarIDSAgent(Agent):
  def __init__(self, compute_action):
    self._compute_action = compute_action
    self._state = 0
    self._rng = None
    
  def reset(self, rng: np.random.Generator, obs_masks: np.ndarray):
    self._obs_masks = obs_masks
    self._num_action = 2
    self._num_state = obs_masks.shape[0]
    self._rng = rng
    self._state = 0
    
    self._num_trans = np.ones(shape=(self._num_state, self._num_action, self._num_state,)) / self._num_state
    self._num_success = np.zeros(shape=(self._num_state, self._num_action,))
    self._num_failure = np.zeros(shape=(self._num_state, self._num_action,))
    self._num_good_obs = np.zeros(shape=(self._num_state, self._num_action,))
    self._num_bad_obs = np.zeros(shape=(self._num_state, self._num_action,))
  
  def update(self, action: int, next_state, obs: np.ndarray):
    
    self._num_trans[self.state, action, next_state] += 1
    self._num_success[self.state, action] += obs[self.state, action]
    self._num_failure[self.state, action] += 1 - obs[self.state, action]
    self._num_good_obs += np.nan_to_num(obs)
    self._num_bad_obs += np.nan_to_num(1 - obs)

  def select_action(self, t: int):
    return self._compute_action(self._rng, t, self._state, self._num_trans, self._num_success, self._num_failure,
                                self.num_good_obs, self.num_bad_obs, self._obs_masks)