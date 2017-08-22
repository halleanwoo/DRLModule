import tensorflow as tf
from EasyDRL.sync import sync
from EasyDRL.agent import A3CAgent
import gym
import numpy as np

GLOBAL_EP = 0
MAX_EP_STEP = 400
MAX_GLOBAL_EP = 800
UPDATE_GLOBAL_ITER = 5
GLOBAL_RUNNING_R = []
GAMMA = 0.9



class Worker(object):
    def __init__(self, name, globalAC, env, sess=None, OPT_A=None, OPT_C=None , COORD=None, continu=True):   # globalAC就是global_net, 针对每个 local——net 都要有 global_net
        self.env = gym.make(GAME).unwrapped
        self.name = name
        self.AC = A3CAgent(name, env, sess, globalAC, OPT_A , OPT_C, continu=True)
        self.COORD = COORD
        self.sess = sess

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP    # 定义全局的  和 episode
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []   # transition 的缓存 
        while not self.COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
    # 执行动作得到 transition
            for ep_t in range(MAX_EP_STEP):
                if self.name == 'W_0':   # 设置为只显示 w_0 的图像
                    self.env.render()
                a = self.AC.choose_action_continu(s)
                s_, r, done, info = self.env.step(a)
                done = True if ep_t == MAX_EP_STEP - 1 else False  # 这个游戏并不存在自动的done为True
                r /= 10     # normalize reward 

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    # 如果到达 更新 global的周期 或者 该episode 运行结束
                    if done:
                        v_s_ = 0   # terminal   对应算法图中的 R
                    else:                
                        v_s_ = self.sess.run(self.AC.v, {self.AC.state: s_[np.newaxis, :] })[0, 0]
#                         if self.name == 'W_0':
#                             print("[ ]:" , SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]}))
#                             print("v_s_" , v_s_)
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r 这里是一个翻转输出的操作（-1从后向前输出） ，因为后面的 r+gamma*v 的操作是从头向尾逐步计算的
                        v_s_ = r + GAMMA * v_s_  # 并不是DQN中利用一个神经网络来输出 q_next
                        # *** 这一步其实就是 折扣公式的形式 : r | r + gamma * r |  r+gamma（ r + gamma * r ） ...
                        buffer_v_target.append(v_s_)  # 存储： V(st) , V(st-1) ...  V(s0)    # 巧妙
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.state: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.a3c_local2global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []   # 训练完了进行清零
                    self.AC.a3c_global2local()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)   # 和画图有关 moving average
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1
                    break


if __name__ == '__main__':
    sess = tf.Session()
    GAME = 'Pendulum-v0'
    env = gym.make(GAME)
    sync(sess, env, Worker)