# -----------------------------
# File: Deep Q-Learning Algorithm
# Author: Yiting Xie
# Date: 2018.9.28
# E-mail: 369587353@qq.com
# -----------------------------

import numpy as np
import os
import cv2
import random
import tensorflow as tf
from collections import deque

GAME_WIDTH = 84 # resized frame width
GAME_HEIGHT = 84 # resized frame height
STATE_LENGTH = 4 # number of image channel

INITIAL_EPSILON = 1.0  # starting value of epsilon
FINAL_EPSILON = 0.1  # final value of epsilon
EXPLORATION_STEPS = 1000000 # Number of steps over which the initial value of epsilon is linearly annealed to its final value

TRAIN_VALUE = 4 # the agent selects 4 actions between successive updates
UPDARE_NETWORK_VALUE = 10000 # the frequency with which the target network is updated
SAVE_VALUE = 200000 # the frequency with which the network is saved

REPLAY_SIZE = 20000 # number of steps to populate the replay memory before training starts
REPLAY_MEMORY = 50000 # number of previous transitions to remember

BATCH_SIZE = 32 # size of minibatch
GAMMA = 0.99 # the value of the proportion of study in the past

BASE_NETWORK_PATH = 'saved_networks/'
BASE_SUMMARY_PATH = 'summary/'

class Agent():
    def __init__(self, actions_num, env_name, load_network, agent_model):
        self.actions_num = actions_num # number of action
        self.env_name = env_name # game name
        self.agent_model = agent_model # 'dueldqn' is dueling_DQN, 'duelddqn' is dueling_DDoubleQN
        
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.replay_memory = deque()
        
        self.time_step = 0
        
        # init Q network
        with tf.variable_scope('q_net'):
            self.input_start, self.Q = self.build_network()
            current_scope = tf.get_default_graph().get_name_scope()
            self.qvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=current_scope)
            self.qvars.sort(key = lambda x: x.name)
            
        # init target Q network
        with tf.variable_scope('tq_net'):
            self.input_startT, self.QT = self.build_network()
            current_scope = tf.get_default_graph().get_name_scope()
            self.tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=current_scope)
            self.tvars.sort(key = lambda x: x.name)
        
        self.update_target_q_network = [var[0].assign(var[1]) for var in zip(self.tvars, self.qvars)]
        
        # build training network
        self.action_input, self.q_input, self.loss, self.grads_update = self.build_training_method()
        
        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        
        # make network path
        if not os.path.exists(BASE_NETWORK_PATH + env_name + '/' + agent_model):
            os.makedirs(BASE_NETWORK_PATH + env_name + '/' + agent_model)
        
        # load network
        if load_network:
            self.load_network()

        # init target network
        self.sess.run(self.update_target_q_network)
        
        # init summary parameters
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(BASE_SUMMARY_PATH + env_name + '/' + agent_model, self.sess.graph)
        
    def build_network(self):   
        input_start = tf.placeholder(tf.float32, [None, GAME_WIDTH, GAME_HEIGHT, STATE_LENGTH])
        conv1 = tf.layers.conv2d(input_start, filters = 32, kernel_size = (8, 8), strides = (4, 4), activation = tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, filters = 64, kernel_size = (4, 4), strides = (2, 2), activation = tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, filters = 64, kernel_size = (3, 3), strides = (1, 1), activation = tf.nn.relu)
        conv3_flatten = tf.contrib.layers.flatten(conv3)
        # advantage stream
        fc1_a = tf.layers.dense(conv3_flatten, 512, activation = tf.nn.relu)
        advantage = tf.layers.dense(fc1_a, self.actions_num, activation = tf.nn.relu)
        
        # value stream
        fc1_v = tf.layers.dense(conv3_flatten, 512, activation = tf.nn.relu)
        value = tf.layers.dense(fc1_v, 1, activation = tf.nn.relu)
        
        # Q Value layer  
        Q = value + (advantage - tf.reduce_mean(advantage, axis = 1, keepdims = True))
        
        return input_start, Q
    
    def build_training_method(self):
        action_input = tf.placeholder(tf.int64, [None])
        q_input = tf.placeholder(tf.float32, [None])
        
        action_onehot = tf.one_hot(action_input, self.actions_num, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.multiply(self.Q, action_onehot), axis = 1)
                
        loss = tf.losses.huber_loss(q_input, q_value)
        optimizer = tf.train.RMSPropOptimizer(0.00025, momentum = 0.95, epsilon =  0.01)
        grads_update = optimizer.minimize(loss)

        return action_input, q_input, loss, grads_update

    def initial_state(self, observation, last_observation):
        new_observation = np.maximum(observation, last_observation)
        gray_observation = cv2.resize(cv2.cvtColor(new_observation, cv2.COLOR_BGR2GRAY),(GAME_WIDTH, GAME_HEIGHT),interpolation = cv2.INTER_CUBIC) * 255
        state = [np.uint8(gray_observation) for _ in range(STATE_LENGTH)]
        return np.stack(state, axis=2)
    
    def get_action(self, state):
        if self.epsilon >= random.random() or self.time_step < REPLAY_SIZE:
            action = random.randrange(self.actions_num)
        else:
            action = np.argmax(self.Q.eval(feed_dict = {self.input_start: [np.float32(state / 255)]}))
        
        # anneal epsilon linearly over time
        if self.epsilon > FINAL_EPSILON and self.time_step >= REPLAY_SIZE:
            self.epsilon -= self.epsilon_step
            
        return action
    
    def get_action_test(self, state):
        if random.random() <= 0.05:
            action = random.randrange(self.actions_num)
        else:
            action = np.argmax(self.Q.eval(feed_dict={self.input_start: [np.float32(state / 255.0)]}))

        self.time_step += 1

        return action
    
    def run_agent(self, action, reward, game_state, next_observation, state):
        next_state = np.append(state[: ,:, 1:], next_observation, axis = 2)
        reward = np.clip(reward, -1, 1)
        self.replay_memory.append((state, action, reward, next_state, game_state))
        
        if len(self.replay_memory) > REPLAY_MEMORY:
            self.replay_memory.popleft()
        
        if self.time_step >= REPLAY_SIZE:
            # train network every TRAIN_VALUE iteration
            if self.time_step % TRAIN_VALUE == 0:
                self.train_network()
            
            # update network every UPDARE_NETWORK_VALUE iteration
            if self.time_step % UPDARE_NETWORK_VALUE == 0:
                self.sess.run(self.update_target_q_network)
            
            # save network every SAVE_VALUE iteration
            if self.time_step % SAVE_VALUE == 0:
                save_path = self.saver.save(self.sess, BASE_NETWORK_PATH + self.env_name + '/' + self.agent_model + '/' + 'game_model', global_step = self.time_step)
                print('Successfully saved: ' + save_path)
        
        self.time_step += 1
        
        # Summary
        self.total_reward += reward
        self.total_q_max += np.max(self.Q.eval(feed_dict={self.input_start: [np.float32(state / 255.0)]}))
        self.duration += 1
        if game_state:
            # Write summary
            if self.time_step >= REPLAY_SIZE:
                stats = [self.total_reward, self.total_q_max / float(self.duration),
                        self.duration, self.total_loss / (float(self.duration) / float(TRAIN_VALUE))]
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode + 1)
            # debug and print info
            if self.time_step <= REPLAY_SIZE:
                mode = 'random'
            elif REPLAY_SIZE <= self.time_step <= REPLAY_SIZE + EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'
            print('AGENT_MODEL:{0} ---- EPISODE: {1:6d} / TIMESTEP: {2:8d} / DURATION: {3:5d} / EPSILON: {4:.5f} / TOTAL_REWARD: {5:3.0f} / AVG_MAX_Q: {6:2.4f} / AVG_LOSS: {7:.5f} / MODE: {8}'.format(
                self.agent_model,
                self.episode + 1, self.time_step, 
                self.duration, self.epsilon,
                self.total_reward, self.total_q_max / float(self.duration),
                self.total_loss / (float(self.duration) / float(TRAIN_VALUE)), mode))
            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1
        
        return next_state
    
    # train q network
    def train_network(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        terminal_batch = [data[4] for data in minibatch]
        # convert true/false to 1/0
        terminal_batch = np.array(terminal_batch) + 0
        
        # Step 2: calculate q value
        q_batch = []
        if self.agent_model == 'duelddqn':
            next_action_batch = np.argmax(self.Q.eval(feed_dict = {self.input_start: np.float32(np.array(next_state_batch) / 255.0)}), axis=1)
            target_q_batch = self.QT.eval(feed_dict = {self.input_startT: np.float32(np.array(next_state_batch) / 255.0)})
            for i in range(BATCH_SIZE):
                q_batch.append(reward_batch[i] + (1 - terminal_batch[i]) * GAMMA * target_q_batch[i][next_action_batch[i]])
        else:
            target_q_batch = self.QT.eval(feed_dict = {self.input_startT: np.float32(np.array(next_state_batch) / 255.0)})
            for i in range(BATCH_SIZE):
                q_batch.append(reward_batch[i] + (1 - terminal_batch[i]) * GAMMA * np.max(target_q_batch[i]))
        
        loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict = {
            self.input_start: np.float32(np.array(state_batch) / 255.0),
            self.action_input: action_batch,
            self.q_input: q_batch
        })
        
        self.total_loss += loss
    
    # load network    
    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(BASE_NETWORK_PATH + self.env_name + '/' + self.agent_model)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')
    
    # setup summary
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar(self.env_name + '/Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar(self.env_name + '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar(self.env_name + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar(self.env_name + '/Average Loss/Episode', episode_avg_loss)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

def process_observation(observation, last_observation):
    new_observation = np.maximum(observation, last_observation)
    gray_observation = cv2.resize(cv2.cvtColor(new_observation, cv2.COLOR_BGR2GRAY), (GAME_WIDTH, GAME_HEIGHT), interpolation = cv2.INTER_CUBIC) * 255
    return np.reshape(np.uint8(gray_observation), (GAME_WIDTH, GAME_HEIGHT, 1))
