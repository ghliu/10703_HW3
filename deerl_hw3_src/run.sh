#!/bin/bash

# problem 1 (LQR)
python3 pr1.py --controller lqr --env TwoLinkArm-v0
python3 pr1.py --controller lqr --env TwoLinkArm-limited-torque-v0
python3 pr1.py --controller lqr --env TwoLinkArm-v1
python3 pr1.py --controller lqr --env TwoLinkArm-limited-torque-v1

# problem 1 (iLQR)
python3 pr1.py --controller ilqr --env TwoLinkArm-v0
python3 pr1.py --controller ilqr --env TwoLinkArm-v1

# problem 1 (extra credit: Speedup & MPC)
python3 pr1.py --controller ilqr-speedup --env TwoLinkArm-v0
python3 pr1.py --controller ilqr-speedup --env TwoLinkArm-v1
python3 pr1.py --controller mpc --env TwoLinkArm-v0
python3 pr1.py --controller mpc --env TwoLinkArm-v1

# problem 2 (Imitation module, include extra credit on DAGGER)
python3 pr2.py -n_ep 1 > ./output/pr2_1
python3 pr2.py -n_ep 10 > ./output/pr2_10
python3 pr2.py -n_ep 50 > ./output/pr2_50
python3 pr2.py -n_ep 100 > ./output/pr2_100

# problem 3 (REINFORCE)
python3 pr3.py -type linear > ./output/pr3_linear
python3 pr3.py -type default > ./output/pr3_default
