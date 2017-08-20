import gym
# import collections
from EasyRL.agent1 import DQNAgent
import tensorflow as tf
import random
import numpy as np
from EasyRL.memory import Memory

MEMORY_SIZE = 10000

EPISODES = 500
MAX_STEP = 500
BATCH_SIZE = 32
UPDATE_PERIOD = 500  # update target network parameters


ENV = "CartPole-v0"

def train( agent , env ):
    # # memory for momery replay
    # memory = []
    # Transition = collections.namedtuple("Transition" , ["state", "action" , "reward" , "next_state" , "done"])
    memory = Memory(MEMORY_SIZE)
    
    reward_his = []
    all_reward = 0
    step_his = []
    update_iter = 0
    for episode in range(EPISODES):
        state = env.reset()
#         env.render() 
        #training
        for step in range(MAX_STEP):
            action = agent.choose_action(state)
            next_state , reward , done , _ = env.step(action)
            all_reward += reward 

# # todo ---完成重构，需要进行替换
#             if len(memory) > MEMORY_SIZE:
#                 memory.pop(0)
#             memory.append(Transition(state, action , reward , next_state , float(done)))

            memory.add(state, action , reward , next_state , float(done))
# todo 
            if len(memory) > BATCH_SIZE * 4:
                # batch_transition = random.sample(memory , BATCH_SIZE)
                # batch_state, batch_action, batch_reward, batch_next_state, batch_done = map(np.array , zip(*batch_transition))  
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = memory.sample(BATCH_SIZE)

                agent.train(state = batch_state ,
                          reward = batch_reward , 
                          action = batch_action , 
                          state_next = batch_next_state,
                          done = batch_done
                         )
                update_iter += 1

            if update_iter % UPDATE_PERIOD == 0:
                agent.update_prmt()

            if update_iter % 200 == 0:
                agent.decay_epsilon()

            if done:
                step_his.append(step)
                reward_his.append(all_reward)
                print("[episode= {} ] step = {}".format(episode , step))
                break

            state = next_state
            
    loss_his = agent.loss_his
    q_value_his = agent.q_value_his
    q_target_his = agent.q_target_his
    q_his = agent.q_his
    qq = agent.qq
    return [step_his , reward_his , loss_his , q_value_his , q_target_his , q_his , qq]


env = gym.make(ENV)

with tf.Session() as sess:
    with tf.variable_scope("DQN"):
        DQN = DQNAgent( "DQN" , env , sess = sess , double = False , out_graph = False , out_dqn = True )
    with tf.variable_scope("Double"):
        Double = DQNAgent("Double" , env , sess = sess , double = True , out_graph = False , out_dqn = False )
        
    sess.run(tf.global_variables_initializer())   
       
    step_double , reward_double , loss_double , q_value_double , q_target_double , q_double , qq_double = train(Double , env)
    step_dqn , reward_dqn , loss_dqn , q_value_dqn , q_target_dqn , q_dqn , qq_dqn = train(DQN , env)