from deeprl_hw3 import imitation
import gym
import argparse

expert = imitation.load_model('CartPole-v0_config.yaml', 'CartPole-v0_weights.h5f')
env = gym.make('CartPole-v0')

parser = argparse.ArgumentParser()
parser.add_argument("-n_ep", "--num_episodes", dest = "num_episodes" , default = 100, help = "Input the number of training episodes from expert.")

args = parser.parse_args()

print("====== Problem 2.1 =====")
# 2.1
observations, expert_actions = imitation.generate_expert_training_data(expert, env, num_episodes = int(args.num_episodes), render = False)
model = imitation.load_model('CartPole-v0_config.yaml')
imitation.behavior_cloning(model, observations, expert_actions)

print("===== Problem 2.2 =====")
# 2.2
imitation.test_cloned_policy(env, model, render = False)

print("===== Problem 2.3 =====")
# 2.3
harder_env = imitation.wrap_cartpole(env)
print("----- evaluate our cloned model -----")
imitation.test_cloned_policy(harder_env, model, render = False)
print("----- evaluate our expert model -----")
imitation.test_cloned_policy(harder_env, expert, render = False)
