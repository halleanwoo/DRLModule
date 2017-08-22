import tensorflow as tf
import numpy as np
import EasyDRL.net_frame as net_frame

# built class for the agent of DQN

class Agent(object):
    def __init__(self, 
                 scope_main, 
                 env, 
                 dueling=False, 
                 double=False, 
                 sess=None, 
                 gamma=0.8, 
                 epsilon=0.8,  
                 out_graph=False, 
                 out_dqn=True, 
                 clip_norm=None,
                 continu=False,
                 ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.loss_his = []
        self.q_value_his = []
        self.q_target_his = []
        self.q_his = []
        self.qq = []
        
        self.state_dim = env.observation_space.shape[0]
        if continu:
            self.action_dim = env.action_space.shape[0]
            self.bound_low = env.action_space.low
            self.bound_high = env.action_space.high
        else:
            self.action_dim = env.action_space.n
        

        self.scope_main = scope_main
        self.dueling = dueling
        self.double = double
        self.out_dqn = out_dqn
        self.clip_norm = clip_norm

    # ===============choose action===================  
    # ---------------discrete action-----------------
    # choose discrete action
    def choose_action(self , current_state):
        current_state = current_state[np.newaxis , :]  #*** array dim: (xx,)  --> (1 , xx) ***
        q = self.sess.run(self.q_value , feed_dict={self.inputs_q : current_state} )        
        
        # e-greedy
        if np.random.random() < self.epsilon:
            action_chosen = np.random.randint(0 , self.action_dim)
        else:
            action_chosen = np.argmax(q)
        
        return action_chosen

    def greedy_action(self , current_state):
        current_state = current_state[np.newaxis , :]  
        q = self.sess.run(self.q_value , feed_dict={self.inputs_q : current_state} ) 
        action_greedy = np.argmax(q)
        return action_greedy

    #  -----------continuous action------------------
    # continuous action normal distribution
    def action_continu_norm_dist(self, mu, sigma, sigma_noise=1e-4):
         mu = mu * self.bound_high
         sigma = sigma + sigma_noise
         norm_dist = tf.contrib.distributions.Normal(mu, sigma)
         self.action = tf.clip_by_value(tf.squeeze(norm_dist.sample(1), axis=0), self.bound_low, self.bound_high)
         return norm_dist

    def choose_action_continu(self, current_state):
        current_state = current_state[np.newaxis , :] 
        action = self.sess.run(self.action, feed_dict={self.state: current_state})[0]
        return action
         
    # ===========update parmerters=============
    # ----------------dqn---------------------
    # upadate parmerters
    def update_prmt(self):
        q_prmts = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES ,  self.scope_main + "/q_network"  )
        target_prmts = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, self.scope_main + "/target_network"  )
        self.sess.run( [tf.assign(t , q)for t,q in zip(target_prmts , q_prmts)])  #***
        # print("updating target-network parmeters...")

    # ----------------a3c---------------------
    # a3c: apply gradient & update parm   
    def a3c_local2global(self, feed_dict):  
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict) 

    def a3c_global2local(self):  
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    # ============change epsilon===============
    def decay_epsilon(self):
        if self.epsilon > 0.03:
            self.epsilon = self.epsilon - 0.02
        

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
                 continu = False,
                 ):
        # self.gamma = gamma
        # self.epsilon = epsilon
        
        # self.action_dim = env.action_space.n
        # self.state_dim = env.observation_space.shape[0]

        # self.scope_main = scope_main
        # self.dueling = dueling
        # self.double = double
        # self.out_dqn = out_dqn
        # self.clip_norm = clip_norm
        
        # self.network()
        # self.sess = sess
        # tf.summary.FileWriter("DoubleDQN/summaries" , self.sess.graph )
        super().__init__(scope_main, env, dueling, double, sess, gamma, epsilon, out_graph, out_dqn, clip_norm, continu)
        self._network(scope_main)
        self.sess = sess
        tf.summary.FileWriter("DoubleDQN/summaries" , self.sess.graph )

    # create q_network & target_network     
    def _network(self, scope_main):
        with tf.variable_scope(scope_main):
            # q_network
            self.inputs_q = tf.placeholder(dtype = tf.float32 , shape = [None , self.state_dim] , name = "inputs_q")
            scope_var = "q_network"
            self.q_value = net_frame.mlp_frame(64 , self.inputs_q , self.action_dim , scope_var ,  self.dueling,  20, [20] ) #, reuse = True
                
            # target_network
            self.inputs_target = tf.placeholder(dtype = tf.float32 , shape = [None , self.state_dim] , name = "inputs_target")
            scope_tar = "target_network"
            self.q_target = net_frame.mlp_frame([64] , self.inputs_target , self.action_dim , scope_tar ,  self.dueling, [20] , [20] )
            
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

                # PriotityBuffer training
                self.ISweight = tf.placeholder( dtype = tf.float32 , shape = [ None , self.action_dim] , name = "wi")
                self.td_error = q_action - self.target
                self.loss_PriBuff = tf.reduce_sum(self.ISweight * tf.square(q_action - self.target))

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


        # training
    def train_pri(self , state , reward , action , state_next , done, batch_ISweight):
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
        
        q_target_best_mask = ( 1.0 - done) * q_target_best
        target = reward + self.gamma * q_target_best_mask
        batch_ISweight = np.stack([batch_ISweight , batch_ISweight] , axis = -1 )
        loss, td_error, _ = self.sess.run([self.loss , self.td_error, self.train_op] ,
                                 feed_dict={self.inputs_q: state , self.target:target , self.action:action, self.ISweight : batch_ISweight ,} ) 
        return td_error
        # self.loss_his.append(loss)



class A3CAgent(Agent):
    def __init__(self, 
                 scope, 
                 env, 
                 sess,
                 global_net=None,
                 OPT_A=None,
                 OPT_C=None, 
                 continu = False,):
        # global_net 
        # only need the network frame without training
        super().__init__(scope_main=scope, 
                         env=env,
                         continu=continu,
                         )
        self.sess = sess
        if scope == "global_net":
            with tf.variable_scope(scope):
                self.state = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name="state")
                self._network()
                # global_parm
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        # local_net
        # actor_critic
        else:
            with tf.variable_scope(scope):
                self.state = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name="state")
                self.a_his = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim], name="action")
                self.v_target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name = "V_target")

                mu, sigma, self.v = self._network()

                with tf.variable_scope("critic_loss"):
                    td_error = tf.subtract(self.v_target, self.v, name="td_error")
                    self.critic_loss = tf.reduce_mean(tf.square(td_error))

                # loss of tha actor & critic
                with tf.variable_scope("action_loss"):
                    norm_dist = self.action_continu_norm_dist(mu, sigma)
                    log_prob = norm_dist.log_prob(self.a_his)
                    exp_v = log_prob * td_error
                    entropy = norm_dist.entropy()
                    self.exp_v = 0.01 * entropy + exp_v
                    self.actor_loss = tf.reduce_mean(-self.exp_v)

                # local_parm
                with tf.name_scope('local_grad'):
                    self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                    self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                    self.a_grads = tf.gradients(self.actor_loss, self.a_params)
                    self.c_grads = tf.gradients(self.critic_loss, self.c_params)

                with tf.name_scope('sync'):
                    # global2local
                    with tf.name_scope('pull'): 
                        self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, global_net.a_params)] 
                        self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, global_net.c_params)]
                    # local2global
                    with tf.name_scope('push'):
                        self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, global_net.a_params))
                        self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, global_net.c_params))

    def _network(self):
        # with tf.variable_scope(scope):
        w_init = tf.random_normal_initializer(0., .1)
        # actor part
        # return mu & sigma to determine action_norm_dist
        scope_var = "actor"
        mu, sigma = net_frame.mlp_frame([200] , self.state , self.action_dim , scope_var, \
                                        activation_fn=tf.nn.relu6, w_init=w_init, activation_fn_v=tf.nn.tanh, \
                                        activation_fn_a=tf.nn.softplus, continu=True)
        # cirtic part
        # return value of the state
        scope_var = "critic"
        v = net_frame.mlp_frame([100], self.state, 1, scope_var, activation_fn=tf.nn.relu6)
        return mu, sigma, v






