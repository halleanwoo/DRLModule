# nn_frame using for creating Q & target network

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import EasyDRL.utils as U


# mlp network frame
def mlp_frame(hiddens, 
              inpt, 
              num_actions, 
              scope=None, 
              dueling=False, 
              hiddens_a=None, 
              hiddens_v=None, 
              activation_fn_v=tf.nn.relu, 
              activation_fn_a=tf.nn.relu, 
              w_init = tf.truncated_normal_initializer(0 , 0.3),
              continu = False,
              reuse=None,
              ):
# if continu:     activation_fn_v is used as activation_mu
#                 activation_fn_a is used as activation_sigma
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt  
        hiddens = U.is_list(hiddens)
        for hidden in hiddens:
            out = layers.fully_connected(out,  num_outputs=hidden, weights_initializer=w_init, activation_fn=tf.nn.relu)
        if not continu:
            if dueling:
                hiddens_a, hiddens_v = U.dueling_has_hiddens(hiddens_a, hiddens_v)
                q_out = _dueling_frame(hiddens_a, hiddens_v, out, num_actions, activation_fn_v, activation_fn_a, reuse=None)
            else:
                with tf.name_scope("out"):
                    q_out = layers.fully_connected(out, num_outputs=num_actions, weights_initializer=w_init, activation_fn=None) 
            return q_out
        else:
            with tf.name_scope("action_dist_out"):
                mu, sigma = _action_norm_dist(out, num_actions, w_init, activation_fn_v, activation_fn_a)
            return mu,sigma


# cnn network frame
def cnn_frame(hiddens,
               kerners, 
              strides, 
              inpt, 
              num_actions, 
              scope=None, 
              dueling=False, 
              hiddens_a=None, 
              hiddens_v=None, 
              activation_fn_v=tf.nn.relu, 
              activation_fn_a=tf.nn.relu, 
              w_init = tf.truncated_normal_initializer(0 , 0.3),
              continu = False,
              reuse=None,
              ):
# if not dueling: hiddens_a is using to construct the fully-connected layers
# if continu:     activation_fn_v is used as activation_mu
#                 activation_fn_a is used as activation_sigma
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        hiddens = U.is_list(hiddens)
        for kerner, stride in kerners, strides:
            out = tf.nn.conv2d(input=out, filter=kerner, stride=stride)
        out = layers.flatten(out)
        if not continu:
            if dueling:
                hiddens_a, hiddens_v = U.dueling_has_hiddens(hiddens_a, hiddens_v)
                q_out = _dueling_frame(hiddens_a, hiddens_v, out, num_actions, activation_fn_v, activation_fn_a, reuse=None)	
            else:
                with tf.variable_scope("out"):
                    for hidden in hiddens_a:
                       out = layers.fully_connected(out, num_outputs=hidden, activation_fn=activation_fn_a)    
                    q_out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None) 
            return q_out
        else:
            with tf.name_scope("action_dist_out"):
                mu, sigma = _action_norm_dist(out, num_actions, w_init, activation_fn_v, activation_fn_a)
            return mu,sigma


## lstm network frame
# def lstm_frame():
#     with tf.variable_scope(scope, reuse=reuse):


def _dueling_frame(hiddens_a, 
                   hiddens_v, 
                   inpt, 
                   num_actions, 
                   activation_fn_v, 
                   activation_fn_a, 
                   reuse=None,
                   ):
    # value_stream
    with tf.variable_scope("value_stream"):
        value = inpt
        for hidden in hiddens_v:
            value = layers.fully_connected(value, num_outputs=hidden , activation_fn=activation_fn_v) 
        value = layers.fully_connected(value, num_outputs= 1 , activation_fn=None) 

    # advantage_stream
    with tf.variable_scope("advantage_stream"):
        advantage = inpt
        for hidden in hiddens_a:
            advantage = layers.fully_connected(advantage , num_outputs = hidden , activation_fn=activation_fn_a) 
        advantage = layers.fully_connected(advantage , num_outputs= num_actions , activation_fn=None) 

    # aggregating_moudle
    with tf.variable_scope("aggregating_moudle"):
        q_out = value + advantage - tf.reduce_mean(advantage , axis = 1 , keep_dims = True )  # ***keep_dims
    return q_out


def _action_norm_dist(inpt, num_actions, w_init, activation_fn_v, activation_fn_a):
    mu = layers.fully_connected(inpt, num_outputs=num_actions, weights_initializer=w_init, activation_fn=activation_fn_v)
    sigma = layers.fully_connected(inpt, num_outputs=num_actions, weights_initializer=w_init, activation_fn=activation_fn_a)
    return mu, sigma


 
# # cnn network frame
# def cnn_frame_continu(hiddens, kerners, strides, inpt, num_actions, scope=None, activation_fn=tf.nn.relu, activation_fn_mu=tf.nn.relu, activation_fn_sigma=tf.nn.relu, reuse=None):
#     with tf.variable_scope(scope, reuse=reuse):
#         out = inpt
#         for kerner, stride in kerners, strides:
#             out = tf.nn.conv2d(input=out, filter=kerner, stride=stride)
#         out = layers.flatten(out)
#         with tf.name_scope("out"):
#             mu = layers.fully_connected(out, num_outputs=num_actions, weights_initializer=tf.truncated_normal_initializer(0 , 0.3), activation_fn=None)
#             sigma = layers.fully_connected(out, num_outputs=num_actions, weights_initializer=tf.truncated_normal_initializer(0 , 0.3), activation_fn=tf.nn.softplus)
#         return mu, sigma