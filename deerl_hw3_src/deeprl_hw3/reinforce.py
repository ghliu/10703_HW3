from deeprl_hw3 import imitation
import gym
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def reinforce(env, model):
    """Policy gradient algorithm

    Parameters
    ----------
    env: your environment
    modle: policy with initialized weights

    Returns
    -------
    total_reward: float
    """

    # construct graphs
    opt = tf.train.AdamOptimizer(learning_rate = 0.01)
    
    Gt = tf.placeholder(tf.float32, shape = [None, 1], name = 'Gt')
    At = tf.placeholder(tf.float32, shape = [None, 2], name = 'At')

    target = tf.squeeze(model.output)
    target1 = tf.reduce_sum(tf.multiply(target, At), axis=[1])
    target2 = tf.log(target1)
    target3 = -Gt*target2
    
    grad_step = opt.compute_gradients(target3, model.weights)

    update_weights = opt.apply_gradients(grad_step)

    # start session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) 

    for i in range(10000):
        state_array, action_array, total_rewards_array = generate_episodes(env, model)
        sess.run(update_weights, feed_dict = {model.input: state_array, Gt:total_rewards_array, At:action_array})

        # set weights
        model.set_weights(sess.run(model.weights))

        if i%20 == 0:
            imitation.test_cloned_policy(env, model, render = False)


def generate_episodes(env, model):
    """Generate episodes of states, actions, and total rewards   

    Parameters
    ----------
    env: your environment
    model: policy

    Returns
    -------
    states array
    stochastic actions array
    total rewards array
    """

    # generate episodes with stochastic actions. similar to imitation.py's generate_expert_training_data
    state_array = []
    action_array = []
    total_rewards_array = []
    rewards_array = []

    state = env.reset()
    is_done = False
    time_step = 0

    while not is_done:
        time_step += 1

        state_array.append(state)
        action_prob = model.predict(np.reshape(state, (-1,4)))
        action = np.random.choice(np.arange(len(action_prob[0])), p=action_prob[0]) # stochastic 
        one_hot_vec = np.zeros(2)
        one_hot_vec[action] = 1
        action_array.append(one_hot_vec) 
        state, reward, is_done, _ = env.step(action)
        rewards_array.append(reward)

    for i in range(time_step):
        total_rewards_array.append([np.sum(rewards_array[i:])])

    return np.array(state_array), np.array(action_array), np.array(total_rewards_array)
