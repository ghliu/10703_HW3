from deeprl_hw3 import imitation, reinforce
import gym
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential
from keras.layers import Dense, Activation

parser = argparse.ArgumentParser()
parser.add_argument("-type", "--type_of_network", dest = "net_type" , default = 'linear', help = "Define the model to be lienar model or the provided model.")

args = parser.parse_args()

env = gym.make('CartPole-v0')


if args.net_type == "linear":
    model = Sequential()
    model.add(Dense(2, input_dim=4, use_bias=False))
    model.add(Activation('softmax'))
    model.summary()
else:
    model = imitation.load_model('CartPole-v0_config.yaml')

reinforce.reinforce(env, model)