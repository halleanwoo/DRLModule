import tensorflow as tf
import numpy as np
import collections
import gym
import random
import matplotlib.pyplot as plt


def choose_action_discrete(net, epsilon, current_state):
    current_state = current_state[np.newaxis , :]  #*** array dim: (xx,)  --> (1 , xx) ***
    q = sess.run(net.q_value , feed_dict={net.inputs_q : current_state} )
    # q_his.append(np.max(q))
    
    # e-greedy
    if np.random.random() < epsilon:
        action_chosen = np.random.randint(0 , action_dim)
    else:
        action_chosen = np.argmax(q)
    
    return action_chosen

def greedy_action_discrete(net, current_state):
    current_state = current_state[np.newaxis , :]  
    q = sess.run(net.q_value , feed_dict={inputs_q : current_state} ) 
    action_greedy = np.argmax(q)
    return action_greedy 
     

#upadate parmerters
def update_prmt_dqn(scope_main):
    q_prmts = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES ,  scope_main + "/q_network"  )
    target_prmts = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope_main + "/target_network"  )
    sess.run( [tf.assign(t , q)for t,q in zip(target_prmts , q_prmts)])  #***
    print("updating target-network parmeters...")

def local2global():
	

def global2local():
    

    
