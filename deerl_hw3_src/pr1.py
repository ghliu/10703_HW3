#!/usr/bin/env python3

import numpy as np
import gym
import time
import pickle

import argparse

from deeprl_hw3.controllers import calc_lqr_input
from deeprl_hw3.ilqr import calc_ilqr_input
from deeprl_hw3.mpc import calc_mpc_input

from ipdb import set_trace as debug

parser = argparse.ArgumentParser(description='DRC 10703 Hw3')

parser.add_argument('--controller', default='lqr', type=str, help='')
parser.add_argument('--env', default='TwoLinkArm-v0', type=str, help='')
parser.add_argument('--tN', default=100, type=int, help='')
parser.add_argument('--max_iter', default=100, type=int, help='')
# parser.add_argument('--dt', default=0.0001, type=float, help='')
parser.add_argument('--tol', default=0.0001, type=float, help='tolerance')
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--useLM', dest='useLM', action='store_true')

args = parser.parse_args()

# build env
env = gym.make(args.env)
sim_env = gym.make(args.env)
goal_q = env.goal_q.copy()

# build controller
select_action = {
	'lqr': lambda env, sim_env: calc_lqr_input(
		env,sim_env,debug_flag=args.debug),
	'ilqr': lambda env, sim_env: calc_ilqr_input(
		env,sim_env,debug_flag=args.debug, max_iter=args.max_iter, tN=args.tN, useLM=False),
	'ilqr-speedup': lambda env, sim_env: calc_ilqr_input(
		env,sim_env,debug_flag=args.debug, max_iter=args.max_iter, tN=args.tN, useLM=True),
	'mpc': lambda env, sim_env: calc_mpc_input(
		env,sim_env,debug_flag=args.debug, max_iter=args.max_iter,
		tN=args.tN, useLM=True),
}.get(args.prob)

# start experiment
env.reset()

action_history = []
state_history = []
reward_history = []

while True:
	sim_env.state = env.state

	action = select_action(env, sim_env)

	obs, reward, is_done, meta = env.step(action)
	
	action_history.append(action)
	state_history.append(obs)
	reward_history.append(reward)

	env.render()

	if np.linalg.norm(goal_q-env.position) < args.tol:
		break

pkl = open('result-prob%s-%s.pkl'%(args.prob,args.env),'wb')
pickle.dump(state_history, pkl)
pickle.dump(action_history, pkl)
pickle.dump(reward_history, pkl)
pkl.close()
