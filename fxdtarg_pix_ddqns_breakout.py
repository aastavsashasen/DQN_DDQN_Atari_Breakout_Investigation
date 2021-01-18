
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> OUR DQN AGENT CLASS (to edit methods/ model)
from __future__ import print_function
import os
import gym
from gym import wrappers
import random
import time
import numpy as np
from collections import deque

import keras
from keras.callbacks import History
from keras.models import Sequential
from keras.layers import Dense, Input, Lambda, Conv2D, Flatten
from keras.optimizers import RMSprop
from keras import backend as K
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
# /Users/sashasen/Documents/CS680/bigtest/fxdtarg_pix_cs680_ddqns_breakout.py

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
    return img[::2, ::2]

def preprocess(img):
    return to_grayscale(downsample(img))
  
def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond  = tf.keras.backend.abs(error) < clip_delta
    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
    return tf.where(cond, squared_loss, linear_loss)
  
class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        self.all_losses = []
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.all_losses.append(logs.get('loss'))
        
  
# Deep Q-learning Agent
class CDDQNAgent:
    def __init__(self, state_size, action_size):
        self.history = LossHistory()
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000000)
        self.gamma = 0.99   
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.1
        self.learning_rate = 0.00025
        self.model = self._build_model()
        self.model_target = self._build_model()
        
    def set_epsilon(self, episode):
        if episode < 10000:
          self.epsilon = -0.00009 * episode + 1.0
        else:
          self.epsilon = self.epsilon_min

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Lambda(lambda x: x / 255.0))
        model.add(Conv2D(16, (8, 8), strides=4, activation='relu', padding='valid',
                         input_shape=self.state_size, data_format='channels_first', init='he_normal'))
        model.add(Conv2D(32, (4, 4), strides=2, activation='relu', padding='valid',
                         data_format='channels_first', init='he_normal'))
        model.add(Conv2D(64, (2, 2), strides=1, activation='relu', padding='valid',
                         data_format='channels_first', init='he_normal'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu', init='he_normal'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=RMSprop(lr=self.learning_rate, rho=0.95, epsilon=0.01))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        act_values = self.model.predict(state)  # gives you an array of q_values
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), act_values[0]
        #print("so called act_values: ", act_values)
        return np.argmax(act_values[0]), act_values[0]  # returns action
      
    def update_target_network_weights(self):
        self.model_target.set_weights(self.model.get_weights()) 
      
    def act_best(self, state): 
        if np.random.rand() <= 0.05:
            return random.randrange(self.action_size)
        # Main network decides action
        act_values = self.model.predict(state)  # gives you an array of q_values
        #print("so called act_values: ", act_values)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):  # use the main q_network to decide action. Get q-value for that action from target network
        minibatch = random.sample(self.memory, batch_size)
        for stack, action, reward, next_state, done in minibatch:
            # if final frame in game do:
            q_value_ideal = reward
            # but the q_value array is...
            q_values = self.model.predict(stack)
            #print("q_values: ", q_values)
            if not done: # else do:
                next_stack = np.empty((1, 4, 105, 80))
                next_stack[0][0] = stack[0][1]
                next_stack[0][1] = stack[0][2]
                next_stack[0][2] = stack[0][3]
                next_stack[0][3] = preprocess(next_state)
                # Wish to approach the ideal Q value, computed with next state
                # action selected by main q-network
                action_selected = np.argmax(np.squeeze(self.model.predict(next_stack)))
                #print("q-values predicted by main network: ", np.squeeze(self.model.predict(next_stack)))
                #print("action selected by main network: ", action_selected)
                # q-value taken from target network
                target_q_values = np.squeeze(self.model_target.predict(next_stack))
                #print("q-values predicted by target network: ", np.squeeze(self.model_target.predict(next_stack)))
                #print("thus, ideal q-value selected (target network value, for main network action): ", target_q_values[action_selected])
                q_value_ideal = reward + self.gamma * target_q_values[action_selected]
            #print("Changing the main q-value array from: ",  q_values[0])
            q_values[0][action] = q_value_ideal
            
            self.model.fit(stack, q_values, epochs=1, verbose=0, callbacks=[self.history])
            
print("CDDQN Class Initialized")
# ===============================================================

env = gym.make('BreakoutDeterministic-v4')
#env = wrappers.Monitor(env, "/tmp/CartPole-v0", force=True)

state_size = env.observation_space.shape
print("Original State Size is: ", state_size)
pr_state_size = preprocess(env.observation_space.sample()).shape
print("Processed State Size is: ", pr_state_size)
action_size = env.action_space.n
print("Action Size is: ", action_size)

# As our model processes 4 frames as a time....
# This would be (4, 105, 80), 4 stacks of 150x80
frame_stack_numb = 4
input_shape = (frame_stack_numb, 105, 80)
agent = CDDQNAgent(input_shape, action_size)
# """""""""""""""""""""""""""""""""""""" Test a SINGLE FIT! ++++++++++++++++++++++++++++++++++++++
state = env.reset()
state = preprocess(state)
stacked_states = np.empty((1, 4, 105, 80))
for i in range(4):
  stacked_states[0][i] = state
print("Stacked states shape: ", stacked_states.shape)
Q_action_values = np.empty((1, 4))
print("Q-values: ", Q_action_values)
# Have to fit the model before running model.summary()
agent.model.fit(stacked_states, Q_action_values, epochs=1, verbose=1)
print(agent.model.summary())
print("Array of q_values: ", np.squeeze(agent.model.predict(stacked_states)))

# ============== Fill with fake memories
action, _ = agent.act(stacked_states)
print("action array: ", action)
next_state, reward, done, info = env.step(action)
for i in range(100):
  agent.remember(stacked_states, action, reward, next_state, done)

agent.replay(32)

# >>>>>>>>>>>>>>>>>>>>>>>>> CONV. DQN - BREAKOUT V4

def clip_rewards(reward):
    if reward > 0.:
        return 1.
    if reward == 0:
        return 0.
    if reward < 0:
        return -1.

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'algorithm2_model.h5'

# initialize gym environment and the agent
env = gym.make('BreakoutDeterministic-v4')

state_size = env.observation_space.shape
print("Original State Size is: ", state_size)
pr_state_size = preprocess(env.observation_space.sample()).shape
print("Processed State Size is: ", pr_state_size)
action_size = env.action_space.n
print("Action Size is: ", action_size)

# Load saved weights into model
#weights_path = "/content/saved_models/CDQN_1_model.h5"
#agent_test.model.load_weights(weights_path)

# As our model processes 4 frames as a time....
# This would be (4, 105, 80), 4 stacks of 150x80
frame_stack_numb = 4
input_shape = (frame_stack_numb, 105, 80)
agent = CDDQNAgent(input_shape, action_size)
# """""""""""""""""""""""""""""""""""""" Test a SINGLE FIT! ++++++++++++++++++++++++++++++++++++++
state = env.reset()
state = preprocess(state)
stacked_states = np.empty((1, 4, 105, 80))
for i in range(4):
    stacked_states[0][i] = state
Q_action_values = np.empty((1, 4))
print("Q-values: ", Q_action_values)
agent.model.fit(stacked_states, Q_action_values, epochs=1, verbose=0)
print(agent.model.summary())


# =================================================================================
run_timeout = 100000  # only included for application in other games (never reached during training here in Breakout)
episodes = 50000
target_network_update_frequency = 10000  # in terms of frames seen
replay_frequency = 4

# Iterate the game
episode_done_array = np.array([])
time_scores_done_array = np.array([])
rewards_done_array = np.array([])
episode_rewards_array = np.array([])
q_vals_array = np.array([])
# ======================================= TRAINING
# need to stack 4 pre processed frames together
# ie) current and 3 previous. reward, next state, done etc of the current state
start = time.time()
frames_seen = 0
for e in range(episodes):  # an Episode is a new game
    #print("just started an episode: ", e, ", frames seen so far: ", frames_seen, ", memory length: ", len(agent.memory))
    # reset state in the beginning of each game
    agent.set_epsilon(e)
    first_state = env.reset()
    episode_reward = 0
    pre_first_state = preprocess(first_state)
    dim1, dim2 = pre_first_state.shape
    stack_array = np.empty((1, frame_stack_numb, dim1, dim2))
    
    count = 0
    # Now let us work with the variable state (updating it):
    state = first_state
    stack_array[0][0] = pre_first_state
    stack_array[0][1] = pre_first_state
    stack_array[0][2] = pre_first_state
    stack_array[0][3] = pre_first_state

    for time_t in range(run_timeout):
        env.render()
        frames_seen += 1
        pre_state = preprocess(state)
        stack_array[0][0] = stack_array[0][1]
        stack_array[0][1] = stack_array[0][2]
        stack_array[0][2] = stack_array[0][3]
        stack_array[0][3] = pre_state

        action, q_vals = agent.act(stack_array)
        q_vals_array = np.append(q_vals_array, q_vals)
        next_state, reward, done, info = env.step(action)
        count += 1
        episode_reward += reward
        reward = clip_rewards(reward)
        if count > 4:
            agent.remember(stack_array, action, reward, next_state, done)
        state = next_state
          
        if frames_seen % target_network_update_frequency == 0:  # update target network weights....
            #print("frames seen: ", frames_seen, ", target network weights updating...")
            agent.update_target_network_weights()
          
        if len(agent.memory) > 5000 and frames_seen % replay_frequency == 0:
            agent.replay(32)
        
        if done:
            # print the score and break out of the loop
            if e % 1000 == 0:
                time_diff = np.abs(np.round((start-time.time())/60))
                print("{} min : Done with... episode: {}/{}, time score: {}, episode reward: {}, frames seen: {}"
                      .format(time_diff, e, episodes, time_t, episode_reward, frames_seen))
                print("Epsilon Value: ", agent.epsilon)
            episode_done_array = np.append(episode_done_array, e)
            time_scores_done_array = np.append(time_scores_done_array, time_t)
            rewards_done_array = np.append(rewards_done_array, episode_reward)
            break
      
    episode_rewards_array = np.append(episode_rewards_array, episode_reward)
env.close()

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
agent.model.save_weights(model_path)
print('Saved trained model at %s ' % model_path)


np.savetxt("algorithm2_episode_rewards_array.csv", np.asarray(episode_rewards_array), delimiter=",")
np.savetxt("algorithm2_episode_done_array.csv", np.asarray(episode_done_array), delimiter=",")
np.savetxt("algorithm2_time_scores_done_array.csv", np.asarray(time_scores_done_array), delimiter=",")
np.savetxt("algorithm2_rewards_done_array.csv", np.asarray(rewards_done_array), delimiter=",")
np.savetxt("algorithm2_loss.csv", np.asarray(agent.history.all_losses), delimiter=",")
np.savetxt("algorithm2_q_vals_array.csv", np.asarray(q_vals_array), delimiter=",")
print("Session Complete")
# Now to test our model....
# ======================================================= TESTING
print("<<<<<<<<<<<<<<<<<<<< STARTING TESTING >>>>>>>>>>>>>>>>>>>>>>>")
env = gym.make('BreakoutDeterministic-v4')

# =================================================================================

test_episodes = 1000

# Iterate the game
test_episode_rewards_array = np.array([])
# ======================================= TRAINING
# need to stack 4 pre processed frames together
# ie) current and 3 previous. reward, next state, done etc of the current state
start_test = time.time()
frames_seen = 0
for e in range(test_episodes):  # an Episode is a new game

    first_state = env.reset()
    episode_reward = 0
    pre_first_state = preprocess(first_state)
    dim1, dim2 = pre_first_state.shape
    stack_array = np.empty((1, frame_stack_numb, dim1, dim2))

    # Now let us work with the variable state (updating it):
    state = first_state
    stack_array[0][0] = pre_first_state
    stack_array[0][1] = pre_first_state
    stack_array[0][2] = pre_first_state
    stack_array[0][3] = pre_first_state

    while True:
        frames_seen += 1
        env.render()
        pre_state = preprocess(state)
        stack_array[0][0] = stack_array[0][1]
        stack_array[0][1] = stack_array[0][2]
        stack_array[0][2] = stack_array[0][3]
        stack_array[0][3] = pre_state

        action = agent.act_best(stack_array)
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        state = next_state
          
        if done:
            # print the score and break out of the loop
            if e % 50 == 0:
                time_diff = np.abs(np.round((start_test-time.time())/60))
                print("{} min :Game done: test episode: {}/{}, frames seen so far: {}, Episode Reward = {}" .format(time_diff, e, test_episodes, frames_seen, episode_reward))
            test_episode_rewards_array = np.append(test_episode_rewards_array, episode_reward)
            break
        

env.close()

# here a test_episode corresponds to a full game (of 5 lives)
np.savetxt("algorithm2_TEST_episode_rewards_array.csv", np.asarray(test_episode_rewards_array), delimiter=",")


final_score_avg = np.mean(test_episode_rewards_array)
print(" ==================================== ")
print("FINAL TEST SCORE AVG.: ", final_score_avg)

print("Test time: ", np.abs(np.round((start_test-time.time())/60)))
print("Total time: ", np.abs(np.round((start-time.time())/60)))

