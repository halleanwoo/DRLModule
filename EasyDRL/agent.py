import tensorflow as tf
import numpy as np
import EasyDRL.net_frame as net_frame
import EasyDRL.utils as U

# built class for the agent of DQN

class Agent(object):
    def __init__(self, 
                 scope_main = None, 
                 env = None, 
                 dueling = False, 
                 double = False, 
                 sess = None, 
                 gamma = 0.8, 
                 epsilon = 0.8,  
                 out_graph = False, 
                 out_dqn = True, 
                 clip_norm = None,
                 continu = False,
                 state_dim = 0,
                 action_dim = 0,
                 bound_low = 0,
                 bound_high = 0,
                 ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.loss_his = []
        self.q_value_his = []
        self.q_target_his = []
        self.q_his = []
        self.qq = []
        
        self.scope_main = scope_main
        self.dueling = dueling
        self.double = double
        self.out_dqn = out_dqn
        self.clip_norm = clip_norm

        # if env is not None, it is using Gym env directly. 
        if env == None:
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.bound_low = low
            self.bound_high = bound_high
        else:
            self.state_dim = env.observation_space.shape[0]
            if continu:
                self.action_dim = env.action_space.shape[0]
                self.bound_low = env.action_space.low
                self.bound_high = env.action_space.high
            else:
                self.action_dim = env.action_space.n

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
        
    # greedy_action policy
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
         
    # ===========update paramerters=============
    # ----------------dqn---------------------
    # upadate paramerters
    def update_prmt(self):
        q_prmts = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES ,  self.scope_main + "/q_network"  )
        target_prmts = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, self.scope_main + "/target_network"  )
        self.sess.run( [tf.assign(t , q)for t,q in zip(target_prmts , q_prmts)])  #***
        # print("updating target-network parameters...")

    # ----------------a3c---------------------
    # a3c: apply gradient   
    def a3c_local2global(self, feed_dict):  
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict) 
    # a3c: update params
    def a3c_global2local(self):  
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    # ============change epsilon===============
    def decay_epsilon(self):
        if self.epsilon > 0.03:
            self.epsilon = self.epsilon - 0.02
        

# ===============================================================
#                             DQN Agent
#                  (DQN , Double DQN , Dueling DQN)
# ===============================================================
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


    # training with priority buffer replay(under_correcting)
    def train_priority(self , state , reward , action , state_next , done, batch_ISweight):
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


# ===============================================================
#                             A3C Agent
# ===============================================================
class A3CAgent(Agent):
    def __init__(self, 
                 scope, 
                 env, 
                 sess,
                 global_net=None,
                 OPT_A=None,
                 OPT_C=None, 
                 continu = False,):

        super().__init__(scope_main=scope, 
                         env=env,
                         continu=continu,
                         )
        self.sess = sess
        # global_net 
        # only need the network frame without training
        if scope == "global_net":
            with tf.variable_scope(scope):
                self.state = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name="state")
                self._network()
                # global_param
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

                # loss of tha actor & critic
                with tf.variable_scope("critic_loss"):
                    td_error = tf.subtract(self.v_target, self.v, name="td_error")
                    self.critic_loss = tf.reduce_mean(tf.square(td_error))

                with tf.variable_scope("action_loss"):
                    norm_dist = self.action_continu_norm_dist(mu, sigma)
                    log_prob = norm_dist.log_prob(self.a_his)
                    exp_v = log_prob * td_error
                    entropy = norm_dist.entropy()
                    self.exp_v = 0.01 * entropy + exp_v
                    self.actor_loss = tf.reduce_mean(-self.exp_v)

                # local_params & gradients
                with tf.name_scope('local_grad'):
                    self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                    self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                    self.a_grads = tf.gradients(self.actor_loss, self.a_params)
                    self.c_grads = tf.gradients(self.critic_loss, self.c_params)

                with tf.name_scope('sync'):
                    # global2local --- pass the global params to local
                    with tf.name_scope('global2local'): 
                        self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, global_net.a_params)] 
                        self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, global_net.c_params)]
                    # local2global --- apply the local gradient to global , changing the params of global
                    with tf.name_scope('local2global'):
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


# ===============================================================
#                             DDPG Agent
# ===============================================================
class DDPGAgent(Agent):
    def __init__(self, 
                 env = None, 
                 sess = None, 
                 learning_rate_actor = 0.1, 
                 learning_rate_critic = 0.1, 
                 gamma = 0.8, 
                 replace_iter_actor = 500, 
                 replace_iter_critic = 500, 
                 batch_size = 32, 
                 tau=None,
                 ):
        super().__init__(env=env, sess=sess , continu=True)

        self.learning_rate_critic = learning_rate_critic
        self.replace_iter_actor = replace_iter_actor
        self.replace_iter_critic = replace_iter_critic
        self.t_replace_counter = 0

        self.learning_rate_actor = learning_rate_actor
        self.gamma = gamma
        self.sess = sess
        self.tau = tau

        STATE = tf.placeholder(tf.float32, shape=[None, self.state_dim], name='s')
        ACTION = tf.placeholder(tf.float32, shape=[None, self.action_dim], name='a')
        REWARD = tf.placeholder(tf.float32, [None, 1], name='r')
        NEXT_STATE = tf.placeholder(tf.float32, shape=[None, self.state_dim], name='s_')
        global STATE, ACTION, REWARD, NEXT_STATE
        self.actor = Actor(sess, self.action_dim, self.bound_high, self.bound_low, self.learning_rate_actor, self.replace_iter_actor, STATE, ACTION, REWARD, NEXT_STATE, self.tau)
        self.critic = Critic(sess, self.state_dim, self.action_dim, self.learning_rate_critic, self.gamma, self.replace_iter_critic, self.actor.next_action, STATE, ACTION, REWARD, NEXT_STATE, self.tau)
        self.actor.add_grad_to_graph(self.critic.a_grads, batch_size)
        self.sess.run(tf.global_variables_initializer()) 

# -----------------------------------------------------------------------
#                            Actor Part
# -----------------------------------------------------------------------
class Actor(object):
    def __init__(self, 
                 sess, 
                 action_dim, 
                 bound_high, 
                 bound_low,
                 learning_rate, 
                 t_replace_iter, 
                 STATE, 
                 ACTION, 
                 REWARD, 
                 NEXT_STATE,
                 tau = None,
                 ):
        self.sess = sess
        self.action_dim = action_dim
        self.bound_high = bound_high
        self.bound_low = bound_low 
        self.learning_rate = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0
        self.STATE = STATE 
        self.ACTION = ACTION
        self.REWARD = REWARD
        self.NEXT_STATE = NEXT_STATE
        self.tau = tau

        with tf.variable_scope('Actor'):
            # actor_eval_net: input state, output action using to step
            self.action= self._build_net(self.STATE, scope='eval_net', trainable=True)   
            # actor_target_net: input next_state, output next_action using for critic to get Q(next_state, next_action)
            self.next_action = self._build_net(self.NEXT_STATE, scope='target_net', trainable=False)  

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(s, 64, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable) 
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.action_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.bound_high, name='scaled_a')  
        return scaled_a

    def learn(self, state, action):   # batch update
        self.sess.run(self.train_op, feed_dict={self.STATE: state, self.ACTION: action})

        # hard updating policy or soft updating policy
        if self.t_replace_counter % self.t_replace_iter == 0:
            if self.tau is None:
                self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
            else:
                self.sess.run([tf.assign(t, (1 - self.tau) * t + self.tau * e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    # choosing single action with noise
    def choose_action(self, current_state, var):
        current_state = current_state[np.newaxis, :]    # keep dim
        action = self.sess.run(self.action, feed_dict={self.STATE: current_state})[0] 
        action_out = np.clip(np.random.normal(action, var), self.bound_low, self.bound_high) 
        return action_out

    def add_grad_to_graph(self, a_grads, batch_size):
        with tf.variable_scope('policy_grads'):
            # *** determintic policy :   d(q)/d(a) * d(a)/d(e_params)
            self.policy_grads = tf.gradients(ys=self.action, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('actor_train'):
            opt = tf.train.AdamOptimizer(-self.learning_rate / batch_size)  # (- learning rate) for ascent policy, div to take mean
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))

# -----------------------------------------------------------------------
#                            Critic Part
# -----------------------------------------------------------------------
class Critic(object):
    def __init__(self, 
                 sess, 
                 state_dim, 
                 action_dim, 
                 learning_rate, 
                 gamma, 
                 t_replace_iter, 
                 next_action, 
                 STATE,  
                 ACTION, 
                 REWARD, 
                 NEXT_STATE,
                 tau = None,
                 ):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0
        self.STATE = STATE 
        self.ACTION = ACTION
        self.REWARD = REWARD
        self.NEXT_STATE = NEXT_STATE
        self.tau = tau

        with tf.variable_scope('Critic'):
            # critic_eval_net: input state & action, output Q(s,a)
            self.q = self._build_net(self.STATE, self.ACTION, 'eval_net', trainable=True)

            # critic_target_net: input next_state & next_action, output Q(s',a')
            self.q_next = self._build_net(self.NEXT_STATE, next_action, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = self.REWARD + self.gamma * self.q_next

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, ACTION)[0]   #  (None, action_dim)

    def _build_net(self, state, action, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                layer1 = 64
                w1_s = tf.get_variable('w1_s', [self.state_dim, layer1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.action_dim, layer1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, layer1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(state, w1_s) + tf.matmul(action, w1_a) + b1)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, state, action, reward, next_state):
        self.sess.run(self.train_op, feed_dict={self.STATE: state, self.ACTION: action, self.REWARD: reward, self.NEXT_STATE: next_state})

        # hard updating policy or soft updating policy
        if self.t_replace_counter % self.t_replace_iter == 0:
            if self.tau is None:
                self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
            else:
                self.sess.run([tf.assign(t, (1 - self.tau) * t + self.tau * e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1
