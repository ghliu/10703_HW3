#!/usr/bin/env python3

import numpy as np
import gym
import time

import argparse

from deeprl_hw3.controllers import calc_lqr_input
from deeprl_hw3.ilqr import calc_ilqr_input
from deeprl_hw3.mpc import calc_mpc_input

from ipdb import set_trace as debug

parser = argparse.ArgumentParser(description='DRC 10703 Hw3')

parser.add_argument('--prob', default='1', type=str, help='')
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
	'1': lambda env, sim_env: calc_lqr_input(
		env,sim_env,debug_flag=args.debug),
	'2': lambda env, sim_env: calc_ilqr_input(
		env,sim_env,debug_flag=args.debug, max_iter=args.max_iter, tN=args.tN),
	'2-useLM': lambda env, sim_env: calc_ilqr_input(
		env,sim_env,debug_flag=args.debug, max_iter=args.max_iter,
		tN=args.tN, useLM=True),
	'2-mpc': lambda env, sim_env: calc_mpc_input(
		env,sim_env,debug_flag=args.debug, max_iter=args.max_iter,
		tN=args.tN, useLM=True),
}.get(args.prob)

# start experiment
env.reset()
while True:
	sim_env.state = env.state

	action = select_action(env, sim_env)

	obs, reward, is_done, _ = env.step(action)
	
	env.render()

	if np.linalg.norm(goal_q-env.position) < args.tol:
		break

debug()