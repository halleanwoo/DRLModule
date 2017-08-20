import collections
import random
import numpy as np

class Memory(object):
	def __init__(self, max_size = 10000):
		self._max_size = max_size
		self._memory = []
		self._Transition = collections.namedtuple("Transition", ["state", "action" , "reward" , "next_state" , "done"])

	# @property
	# def size(self):
	# 	self._size = len(self._memory)
	# 	return self._size
	
	def __len__(self):
		return len(self._memory)

	def add(self, state, action, reward, next_state, done):
		transition = self._Transition(state, action, reward, next_state, done)
		if len(self._memory) > self._max_size:
			self._memory.pop(0)
		self._memory.append(transition)

	def sample(self, batch_size = 32):
		if (batch_size > self._max_size or batch_size <= 0):
			raise ValueError("invalid value: batch_size should between (0, memory_size] ")
		batch_transitions = random.sample(self._memory, batch_size)
		batch_state, batch_action, batch_reward, batch_next_state, batch_done = map(np.array , zip(*batch_transitions))
		return batch_state, batch_action, batch_reward, batch_next_state, batch_done

