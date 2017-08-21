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



# ====================correcting==============================================

# proportional variant / used in class Memory
class _SumTree():
    def __init__(self , capcity=2000, alpha=0.5 ):
        self.capcity = capcity
        self.alpha = alpha
    
        # struct of SumTree & memory for the transition
        self.tree = np.zeros( 2 * self.capcity - 1)
        self.data = [None] * self.capcity
        
        # pointer for the position
        self.pointer = 0
    
    # add new priority in leaf_node
    def add_leaf_node(self , p_alpha , transition):
        leaf_idex = self.pointer + self.capcity - 1
                
        self.data[self.pointer] = transition
        self.update_leaf_node(leaf_idex , p_alpha)
    
        self.pointer += 1
        if self.pointer >= self.capcity: # ！not self.capcity-1 
            self.pointer = 0
    
    # update leaf_node according leaf_idex
    def update_leaf_node(self , leaf_idex , p_alpha):
        change = p_alpha - self.tree[leaf_idex]
        self.tree[leaf_idex] = p_alpha 
        self._update_parent_node(change , leaf_idex )
            
    # update the value of sum p in parent node
    def _update_parent_node(self , change , child_idex ):
        parent_idex = (child_idex - 1) // 2
        
        self.tree[parent_idex] += change 
#         print(parent_idex)
#         print(change)
        
        if parent_idex != 0:
            self._update_parent_node(change , parent_idex)    
        
    # sampling to get leaf idex and transition
    def sample(self , sample_idex):
        leaf_idex = self._retrieve(sample_idex)
        data_idex = leaf_idex - self.capcity + 1
        
        return [leaf_idex , self.tree[leaf_idex] , self.data[data_idex] ]
        
    # retrieve with O(log n)
    def _retrieve(self , sample_idex , node_idex = 0):
        left_child_idex = node_idex * 2 + 1
        right_child_idex = left_child_idex + 1
        
        if left_child_idex >= len(self.tree):  # ! must be  >= 
            return node_idex
        
        if self.tree[left_child_idex] == self.tree[right_child_idex]:  
            return self._retrieve(sample_idex , np.random.choice([left_child_idex , right_child_idex]))
        if self.tree[left_child_idex] > sample_idex:
            return self._retrieve(sample_idex , node_idex = left_child_idex)
        else:
            return self._retrieve(sample_idex - self.tree[left_child_idex] , node_idex = right_child_idex )
        
    # sum of p in root node
    def root_priority(self):
        return self.tree[0]



class PrioritizedMemory():
    def __init__(self , tree_epsilon = 0.01, beta=0.1, capcity=2000):
        self.epsilon = tree_epsilon
        self.p_init = 1. 
        self.beta = beta
        self.beta_change_step = 0.001
        self.capcity = capcity
        
        self.sum_tree = _SumTree() 
        self._Transition = collections.namedtuple("Transition", ["state", "action" , "reward" , "next_state" , "done"])
        
    # store transition & priority before replay
      # 直接将新获得的transition以 最大优先级 进行存储，方便 提取出来训练，之后根据td_error进行优先级更新
    def store(self , state, action , reward , next_state , done):
        transition = self._Transition(state, action , reward , next_state , done)
        p_max = np.max(self.sum_tree.tree[-self.capcity:])
        if p_max == 0:
            p_max = self.p_init
        self.sum_tree.add_leaf_node(p_max , transition)
        
    # update SumTree
    def update(self , leaf_idex , td_error  ):
        p = np.abs(td_error) + self.epsilon
        p_alpha = np.power(p , ALPHA)
        
        for i in range(len(leaf_idex)):
            self.sum_tree.update_leaf_node(leaf_idex[i] , p_alpha[i] )
        
    # sample
    def sampling(self , batch_size ):
        batch_idex = []
        batch_transition = []
        batch_ISweight = []
        
        segment = self.sum_tree.root_priority() / batch_size
        
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            sample_idex = np.random.uniform(low , high)
            idex , p_alpha , transition  = self.sum_tree.sampling(sample_idex)
            prob = p_alpha / self.sum_tree.root_priority()
            batch_ISweight.append( np.power(self.capcity * prob , -self.beta) )
            batch_idex.append( idex )
            batch_transition.append( transition)
#         print(np.min(self.sum_tree.tree[-self.capcity:]) )
#         print("root:" , self.sum_tree.root_priority())
        i_maxiwi = np.power(self.capcity * np.min(self.sum_tree.tree[-self.capcity:]) / self.sum_tree.root_priority() , self.beta)
#         print("maxiwi:" ,i_maxiwi )
#         print("isweight" , batch_ISweight)
#         print(batch_transition)

        
        batch_ISweight = np.array(batch_ISweight) * i_maxiwi 
        
        return batch_idex , batch_transition , batch_ISweight
        
    # change beta
    def change_beta(self):
        self.beta -= self.beta_change_step
        return np.min(1 , self.beta)