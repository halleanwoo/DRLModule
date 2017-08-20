import tensorflow as tf
import numpy as np
import EasyRL.net_frame as net_frame

# built class for the agent of DQN

class Agent(object):
    def __init__(self, sess):
        self.sess = sess

    # chose action
    def choose_action(self , current_state):
        current_state = current_state[np.newaxis , :]  #*** array dim: (xx,)  --> (1 , xx) ***
        q = self.sess.run(self.q_value , feed_dict={self.inputs_q : current_state} )        
        
        # e-greedy
        if np.random.random() < self.epsilon:
            action_chosen = np.random.randint(0 , self.action_dim)
        else:
            action_chosen = np.argmax(q)
        
        return action_chosen

    def choose_action_continu(self, current_state):
        current_state = current_state[np.newaxis , :] 
        action = self.sess.run(self.action, feed_dict={self.inputs_q: current_state})[0]
        return action
         
    #upadate parmerters
    def update_prmt(self):
        q_prmts = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES ,  self.scope_main + "/q_network"  )
        target_prmts = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, self.scope_main + "/target_network"  )
        self.sess.run( [tf.assign(t , q)for t,q in zip(target_prmts , q_prmts)])  #***
        print("updating target-network parmeters...")
        
    def decay_epsilon(self):
        if self.epsilon > 0.03:
            self.epsilon = self.epsilon - 0.02
        
    def greedy_action(self , current_state):
        current_state = current_state[np.newaxis , :]  
        q = self.sess.run(self.q_value , feed_dict={self.inputs_q : current_state} ) 
        action_greedy = np.argmax(q)
        return action_greedy


# 
class DQNAgent(Agent):
    def __init__(self, 
                 scope_main, 
                 env, 
                 dueling = False, 
                 double = False, 
                 sess=None, 
                 gamma = 0.8, 
                 epsilon = 0.8,  
                 out_graph = False, 
                 out_dqn = True, 
                 clip_norm = None,
                 ):

        self.gamma = gamma
        self.epsilon = epsilon
        self.loss_his = []
        self.q_value_his = []
        self.q_target_his = []
        self.q_his = []
        self.qq = []
        
        self.action_dim = env.action_space.n
        self.state_dim = env.observation_space.shape[0]

        self.scope_main = scope_main
        self.dueling = dueling
        self.double = double
        self.out_dqn = out_dqn
        self.clip_norm = clip_norm
        
        self.network()
        self.sess = sess
        tf.summary.FileWriter("DoubleDQN/summaries" , self.sess.graph )
        
    # create q_network & target_network     
    def network(self):
        # q_network
        self.inputs_q = tf.placeholder(dtype = tf.float32 , shape = [None , self.state_dim] , name = "inputs_q")
        scope_var = "q_network"
        self.q_value = net_frame.mlp_frame(64 , self.inputs_q , self.action_dim , scope_var , self.dueling, [20] , [20]  ) #, reuse = True
            
        # target_network
        self.inputs_target = tf.placeholder(dtype = tf.float32 , shape = [None , self.state_dim] , name = "inputs_target")
        scope_tar = "target_network"
        self.q_target = net_frame.mlp_frame([64] , self.inputs_target , self.action_dim , scope_tar , self.dueling, [20] , [20] )
        
        # compute loss
        with tf.variable_scope("loss"):
            self.action = tf.placeholder(dtype = tf.int32 , shape = [ None ] , name = "action")
            action_one_hot = tf.one_hot(self.action , self.action_dim )
            q_action = tf.reduce_sum( tf.multiply(self.q_value , action_one_hot) , axis = 1 )
            
            # DoubleDQN
            self.action_best = tf.placeholder(dtype = tf.int32 , shape = [None] , name = "action_best")
            action_best_one_hot = tf.one_hot(self.action_best , self.action_dim)
            self.q_target_action = tf.reduce_sum(tf.multiply(self.q_target , action_best_one_hot) , axis = 1)
            
            self.target =  tf.placeholder(dtype = tf.float32 , shape =  [None ] , name = "target")
            self.loss = tf.reduce_mean( tf.square(q_action - self.target))

        with tf.variable_scope("train"):
            optimizer = tf.train.RMSPropOptimizer(0.001)
            if self.clip_norm :
                gradients = optimizer.compute_gradients(self.loss)
                for i , (g, v) in enumerate(gradients):
                    if g is not None:
                        gradients[i] = (tf.clip_by_norm(g , 10) , v)
                self.train_op = optimizer.apply_gradients(gradients)
            else:
                self.train_op = optimizer.minimize(self.loss)
    
    # training
    def train(self , state , reward , action , state_next , done):
        q , q_target = self.sess.run([self.q_value , self.q_target] ,
                                     feed_dict={self.inputs_q : state , self.inputs_target : state_next } )
        # DoubleDQN
        if self.double:
            q_next = self.sess.run(self.q_value , feed_dict={self.inputs_q : state_next})
            action_best = np.argmax(q_next , axis = 1)
            q_target_best = self.sess.run(self.q_target_action , feed_dict={self.action_best : action_best,
                                                                            self.q_target : q_target})
        else:
            q_target_best = np.max(q_target , axis = 1)   # dqn
        
        # self.q_value_his.append(q)
        # self.q_target_his.append(q_target_best)
        
        q_target_best_mask = ( 1.0 - done) * q_target_best
        
        target = reward + self.gamma * q_target_best_mask
        
        loss , _ = self.sess.run([self.loss , self.train_op] ,
                                 feed_dict={self.inputs_q: state , self.target:target , self.action:action} ) 
        # self.loss_his.append(loss)

