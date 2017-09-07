import gym
from EasyDRL.agent import DDPGAgent
import tensorflow as tf
import random
import numpy as np
from EasyDRL.memory import Memory


   
def train():
    ENV_NAME = 'Pendulum-v0'
    RENDER = False
    OUTPUT_GRAPH = True 

    MAX_EPISODES = 100
    MAX_EP_STEPS = 400
     
    TAU = 0.01 
    REPLACE_ITER_ACTOR = 500    
    REPLACE_ITER_CRITIC = 300
    MEMORY_SIZE = 7000
    BATCH_SIZE = 32
    env = gym.make(ENV_NAME)
    sess = tf.Session()
    memory = Memory(MEMORY_SIZE)

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph) 

    agent = DDPGAgent(env, sess, 0.01, 0.01, 0.9, REPLACE_ITER_ACTOR, REPLACE_ITER_CRITIC, BATCH_SIZE)
    var = 3 

    for i in range(MAX_EPISODES):
        state = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()
            action = agent.actor.choose_action(state, var)
            # print(action)
            next_state, reward, done, _ = env.step(action)

            memory.add(state, action, reward / 10, next_state, 0)

            if len(memory) > MEMORY_SIZE:
                var *= .9995    # decay the action randomness
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = memory.sample(BATCH_SIZE)
                batch_reward = batch_reward.reshape([BATCH_SIZE, 1])

                agent.critic.learn(batch_state, batch_action, batch_reward, batch_next_state)
                agent.actor.learn(batch_state, batch_action)

            state = next_state
            ep_reward += reward

            if j == MAX_EP_STEPS-1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore Ration: %.2f' % var, )
                if ep_reward > -1000:
                    RENDER = True
                break

if __name__ == "__main__":
    train()





















# actor = DPGAgent( env, sess, learning_rate_actor, REPLACE_ITER_A, STATE, ACTION, REWARD, NEXT_STATE)   #"actor",     self.STATE, self.ACTION, self.REWARD, self.NEXT_STATE
# critic = Critic( env, sess, learning_rate_critic, gamma, REPLACE_ITER_C, actor.action, STATE, ACTION, REWARD, NEXT_STATE)  # "critic",      self.STATE, self.ACTION, self.REWARD, self.NEXT_STATE
# actor.add_grad_to_graph(critic.a_grads, batch_size)







# sess.run(tf.global_variables_initializer()) 

# # agent = DDPGAgent(sess, env, LR_A, LR_C, GAMMA, REPLACE_ITER_C, BATCH_SIZE)
# # # Create actor and critic.，
# # # They are actually connected to each other, details can be seen in tensorboard or in this picture:
# # actor = PGAgent(sess, action_dim, action_bound, LR_A, REPLACE_ITER_A)   # REPLACE_ITER_A隔多少步提升target network
# # critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACE_ITER_C, actor.a_)
# #         # 因为ctitic是要检测actor的动作的，所以要接受从actor的 target network 中传过来的动作 actor.a
# # actor.add_grad_to_graph(critic.a_grads)  # actor也接受从critic传过来的 动作梯度  ，即actor更新公式中的 delta_a(Q)那前半部分

# # sess.run(tf.global_variables_initializer())
# memory = Memory(MEMORY_CAPACITY)
# # M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)

# if OUTPUT_GRAPH:
#     tf.summary.FileWriter("logs/", sess.graph)  # 写上sess.graph可以在TensorBoard看 graph

# var = 3  # control exploration

# for i in range(MAX_EPISODES):
#     s = env.reset()
#     ep_reward = 0

#     for j in range(MAX_EP_STEPS):

#         if RENDER:
#             env.render()

#         # Added exploration noise
#         a = actor.choose_action_determin(s,var)
#         # a = np.clip(a, -2, 2)    # add randomness to action selection for exploration
#         s_, r, done, info = env.step(a)

#         memory.add(s, a, r / 10, s_, 0)

#         if len(memory) > MEMORY_CAPACITY:
#                 var *= .9995    # decay the action randomness
#                 # batch_transition = random.sample(memory , BATCH_SIZE)
#                 # batch_state, batch_action, batch_reward, batch_next_state, batch_done = map(np.array , zip(*batch_transition))  
#                 batch_state, batch_action, batch_reward, batch_next_state, batch_done = memory.sample(BATCH_SIZE)
#                 # print(batch_action)
#                 # print("************")
#                 # print(batch_reward)
#                 batch_reward = batch_reward.reshape([32, 1])

#                 critic.learn(batch_state, batch_action, batch_reward, batch_next_state)
#                 actor.learn(batch_state, batch_action)

#         # if M.pointer > MEMORY_CAPACITY:
#         #     
#         #     b_M = M.sample(BATCH_SIZE)
#         #     b_s = b_M[:, :state_dim]
#         #     b_a = b_M[:, state_dim: state_dim + action_dim]
#         #     b_r = b_M[:, -state_dim - 1: -state_dim]
#         #     b_s_ = b_M[:, -state_dim:]

#         s = s_
#         ep_reward += r

#         if j == MAX_EP_STEPS-1:
#             print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
#             if ep_reward > -1000:
#                 RENDER = True
#             break