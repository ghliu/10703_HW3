
import numpy as np
from deeprl_hw3.ilqr import solve


U = None
t = -1
def calc_mpc_input(env, sim_env, tN=50, 
    max_iter=1e6, useLM=False, debug_flag=False):
    """Calculate the optimal control input for the given state.


    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.
    tN: number of control steps you are going to execute
    max_itr: max iterations for optmization

    Returns
    -------
    U: np.array
      The SEQUENCE of commands to execute. The size should be (tN, #parameters)
    """
    global U
    global t

    sim_env.state = env.state.copy()
    action_dim = env.action_space.shape[0] 
    state_dim = env.observation_space.shape[0]

    if U is None:
        U = np.zeros((tN,action_dim))

    # update t
    t = np.mod(t+1,tN)

    # solve
    x0 = sim_env.state.copy()
    UU = np.copy(U[t:])
    X, U[t:], cost = solve(sim_env, x0, UU, max_iter, useLM, debug_flag)

    return U[t]