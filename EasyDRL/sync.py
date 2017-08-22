from EasyDRL.agent import A3CAgent
import tensorflow as tf
import multiprocessing
import threading

class Worker(object):
	def __init__(self, name, )



def sync(sess, 
		 worker,
		 learning_rate_a,
		 learning_rate_c,
		 ):
	n_workers = multiprocessing.cpu_count()
	with tf.device("/cpu:0"):
		OPT_A = tf.train.RMSPropOptimizer(learning_rate_a)
		OPT_C = tf.train.RMSPropOptimizer(learning_rate_c)
		GLOBAL_NET = A3CAgent("global_net")
		workers = []
		for i in range(n_workers):
			i_name = "W_%i" % i
			workers.append(worker)
