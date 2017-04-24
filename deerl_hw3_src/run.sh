#!/bin/bash

python3 pr2.py -n_ep 1 > ./output/pr2_1
python3 pr2.py -n_ep 10 > ./output/pr2_10
python3 pr2.py -n_ep 50 > ./output/pr2_50
python3 pr2.py -n_ep 100 > ./output/pr2_100

python3 pr3.py -type linear > ./output/pr3_linear
python3 pr3.py -type default > ./output/pr3_default
