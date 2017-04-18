#!/usr/bin/env python3

import numpy as np
import gym
from copy import deepcopy
import time

import argparse

from deeprl_hw3.controllers import calc_lqr_input
from deeprl_hw3.ilqr import calc_ilqr_input


from ipdb import set_trace as debug

parser = argparse.ArgumentParser(description='DRC 10703 Hw3')

parser.add_argument('--prob', default=1, type=int, help='')
parser.add_argument('--env', default='TwoLinkArm-v0', type=str, help='')
parser.add_argument('--dt', default=0.0001, type=float, help='')
parser.add_argument('--tol', default=0.0001, type=float, help='tolerance')
parser.add_argument('--debug', dest='debug', action='store_true')

args = parser.parse_args()


env = gym.make(args.env)
sim_env = gym.make(args.env)

action_dim = env.action_space.shape[0]
state_dim  = env.observation_space.shape[0]
goal_q = env.goal_q.copy()

env.reset()

# debug()
while True:
	sim_env.state = env.state
	# action = 10.*np.random.rand(action_dim)
	action = {
		1:calc_lqr_input(env, sim_env, debug_flag=args.debug),
		2:calc_ilqr_input(env, sim_env),
	}.get(args.prob)

	obs, reward, is_done, _ = env.step(action)

	env.render()

	if np.linalg.norm(goal_q-env.position) < args.tol:
		break
	# time.sleep(0.1)