#@title Graph Environment
import numpy as np

class Graph(object):
	def __init__(self,
				 transitions_deterministic=True,
				 max_length = 2,
                 rewarding_state = None):
		self.allowable_actions = [0,1]
		self.n_actions = len(self.allowable_actions)
		self.n_dim = 2*max_length - 1
		# split = np.array_split(np.arange(2, 2*max_length)-1, -1)
		self.max_length = max_length
		self.rewarding_state = rewarding_state
		reward = np.zeros([self.n_dim, self.n_actions])
		reward[self.rewarding_state, 1] = 1
		self.reward = reward
		self.reset()
	@property
	def num_states(self):
		return self.n_dim

	def pos_to_image(self, x):
		'''latent state -> representation '''
		return x

	def reset(self):
		self.state = 0
		self.done = False
		return np.array([self.state])

	def step(self, action):
		assert action in self.allowable_actions
		assert not self.done, 'Episode Over'
		# Generate observation for all (s,a)
		
		prev_state = self.state
		
		obs = np.copy(self.reward)
		
		# Mask all other (s,a) if not transitioning to informative state
		if not (prev_state == 2*self.max_length-4 and action == 0):
			mask = np.zeros([self.num_states, self.n_actions])
			mask[prev_state, action] = 1
			obs[np.logical_not(mask)] = np.nan
			bad_obs = 1 - obs
			num_success = np.nan_to_num(obs)
			num_failure = np.nan_to_num(bad_obs)

			# bad_obs = (1 - obs)*100
			# obs = obs*100
		else:
			num_success = obs*1000
			num_failure = (1 - obs)*1000
			# obs[2*self.max_length-4, 0] = np.nan
			bad_obs = (1 - obs)*100
			obs *= 100

		

		if self.state == (2*self.max_length-3) or self.state == (2*self.max_length-2):
			self.state = 0
			self.done = True
		else:
			if self.state % 2 == 1:
				self.state = self.state + 2
			else:
				if action == 0:
					self.state = self.state + 2
				else:
					self.state = self.state + 1

		state = self.state

		return self.state, num_success, num_failure, obs, bad_obs, self.done

	def render(self, a=None, r=None, return_arr=False):
		start_state = 1 if self.state == 0 else 0
		state = np.zeros(2*self.max_length-2)
		end_state = 1 if self.state == (2*self.max_length-1) else 0

		if not start_state and not end_state:
			state[self.state-1] = 1

		if return_arr:
			return start_state, state.reshape(2,self.max_length-1, order='F'), end_state
		else:

			print(' ', ' '.join(state.reshape(2,self.max_length-1, order='F')[0].astype(int).astype(str).tolist()), '  ')
			if (a is not None) and (r is not None):
				print(start_state, ' '*((2*(self.max_length-2))+1), end_state, ' (a,r): ', (a,r), '.  If POMDP, End state: ', end_state)
			else:
				print(start_state, ' '*((2*(self.max_length-2))+1), end_state)
			print(' ', ' '.join(state.reshape(2,self.max_length-1, order='F')[1].astype(int).astype(str).tolist()), '  ')
			print('\n')
			# print([start_state], [end_state], state.reshape(2,self.max_length-1, order='F'), )