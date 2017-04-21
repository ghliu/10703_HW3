import matplotlib.pyplot as plt
import sys
import pickle

#pkl = open("result-LQR-TwoLinkArm-v0.pkl", "rb")
pkl = open(sys.argv[1], "rb")
states = pickle.load(pkl)
actions = pickle.load(pkl)
rewards = pickle.load(pkl)
pkl.close()

env_name = sys.argv[1].replace("result-","").replace('.pkl','')

plt.plot(states[0], 'b-')
plt.plot(states[1], 'r-')
plt.savefig("states-iter-line-%s.png"%(env_name))
plt.clf()

plt.plot(states[0], states[1], 'b.')
plt.savefig("states-%s.png"%(env_name))
plt.clf()


plt.ylim([-15,15])
plt.plot(actions[0], 'b-')
plt.plot(actions[1], 'r-')
plt.savefig("actions-iter-line-%s.png"%(env_name))
plt.clf()

plt.plot(rewards, 'b-')
plt.savefig("reward-%s.png"%(env_name))
plt.clf()




