"""LQR, iLQR and MPC."""

from deeprl_hw3.controllers import approximate_A, approximate_B
import numpy as np
import scipy.linalg


def simulate_dynamics_next(env, x, u):
    """Step simulator to see how state changes.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.

    Returns
    -------
    next_x: np.array
    """

    env.dt = dt
    env.state = x.copy()
    x1, _, _, _ = env.step(u)
	return x1
    #xdot = (x1-x) / dt
    #return xdot


def cost_inter(env, x, u):
    """intermediate cost function

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.

    Returns
    -------
    l, l_x, l_xx, l_u, l_uu, l_ux. The first term is the loss, where the remaining terms are derivatives respect to the
    env, corresponding variables, ex: (1) l_x is the first order derivative d l/d x (2) l_xx is the second order derivative
    d^2 l/d x^2
    """

    num_actions = u.shape[0]
    num_states = x.shape[0]

    l = np.sum(u**2)
    l_x = np.zeros(num_states)
    l_xx = np.zeros((num_states, num_states))
    l_u = 2 * u
    l_uu = 2 * np.eye(num_actions)
    l_ux = np.zeros((num_actions, num_states))

    return l, l_x, l_xx, l_u, l_uu, l_ux


def cost_final(env, x):
    """cost function of the last step

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.

    Returns
    -------
    l, l_x, l_xx The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables
    """

    num_states = x.shape[0]
    num_actions = env.action_space.shape[0]

    l_x = np.zeros((num_states))
    l_xx = np.zeros((num_states, num_states))

    wp = 1e4 
    wv = 1e4 

    xy = env.x.copy() # copy? 
    target = env.goal_q.copy()

    xy_err = np.array([xy[0] - target[0], xy[1] - target[1]])
    l = (wp * np.sum(xy_err**2) + wv * np.sum(x[num_actions:num_actions*2]**2))

    l_x[:num_actions] = wp * self.diff(x[:num_actions], env)
    l_x[num_actions:num_actions*2] = (2 * wv * x[num_actions:num_actions*2])

    eps = 1e-4 

    for k in range(num_actions): 
        veps = np.zeros(num_actions)
        veps[k] = eps
        d1 = wp * self.diff(x[0:num_actions] + veps, env)
        d2 = wp * self.diff(x[0:num_actions] - veps, env)
        l_xx[0:num_actions, k] = ((d1-d2) / 2.0 / eps).flatten()

    l_xx[num_actions:num_actions*2, num_actions:num_actions*2] = 2 * wv * np.eye(num_actions)
    return l, l_x, l_xx


def diff(self, x, env):
    num_actions = env.action_space.shape[0]
    target = env.goal_q.copy()

    xe = -target.copy()
    for ii in range(num_actions):
        lii = env.l1 if ii==0 else env.l2
        xe[0] += lii * np.cos(np.sum(x[:ii+1]))
        xe[1] += lii * np.sin(np.sum(x[:ii+1]))

    edot = np.zeros((num_actions,1))
    for ii in range(num_actions):
        lii = env.l1 if ii==0 else env.l2
        edot[ii,0] += (2 * lii * (xe[0] * -np.sin(np.sum(x[:ii+1])) + xe[1] * np.cos(np.sum(x[:ii+1]))))
    edot = np.cumsum(edot[::-1])[::-1][:]

    return edot


    
def simulate(env, x0, U):
    tN = U.shape[0]
    num_states = x0.shape[0]
    dt = env.dt

    X = np.zeros((tN, num_states))
    X[0] = x0
    cost = 0
    for t in range(tN-1):
        X[t+1] = self.simulate_dynamics_next(env, X[t], U[t])
        l,_,_,_,_,_ = self.cost_inter(X[t], U[t])
        cost = cost + dt * l

    l_f,_,_ = self.cost_final(X[-1])
    cost = cost + l_f

    return X, cost

def calc_ilqr_input(env, sim_env, tN=50, max_iter=1e6):
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
    return np.zeros((50, 2))
