"""LQR, iLQR and MPC."""

import numpy as np

import scipy #.linalg.solve_continuous_are as func

from ipdb import set_trace as debug

def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))


def simulate_dynamics(env, x, u, dt=1e-5):
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
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    xdot: np.array
      This is the **CHANGE** in x. i.e. (x[1] - x[0]) / dt
      If you return x you will need to solve a different equation in
      your LQR controller.
    """

    env.dt = dt
    env.state = x.copy()
    x1, _, _, _ = env.step(u)
    
    xdot = (x1-x) / dt
    return xdot


def approximate_A(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate A matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. You will need to perturb this.
    u: np.array
      The command to test.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    A: np.array
      The A matrix for the dynamics at state x and command u.
    """

    state_dim = x.shape[0]
    action_dim = u.shape[0]

    A = np.zeros((state_dim, state_dim)) 
    xs_inc = np.tile(x,(state_dim,1)) + delta * np.eye(state_dim)
    xs_dec = np.tile(x,(state_dim,1)) - delta * np.eye(state_dim)

    for idx, (x_inc, x_dec) in enumerate(zip(xs_inc, xs_dec)):
        # calculate partial differential w.r.t. x
        state_inc = simulate_dynamics(env, x_inc, u.copy(), dt)
        state_dec = simulate_dynamics(env, x_dec, u.copy(), dt)
        A[:, idx] = (state_inc - state_dec) / (2 * delta)

    return A


def approximate_B(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate B matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test.
    u: np.array
      The command to test. You will ned to perturb this.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    B: np.array
      The B matrix for the dynamics at state x and command u.
    """

    state_dim = x.shape[0]
    action_dim = u.shape[0]

    B = np.zeros((state_dim, action_dim)) 
    us_inc = np.tile(u,(action_dim,1)) + delta * np.eye(action_dim)
    us_dec = np.tile(u,(action_dim,1)) - delta * np.eye(action_dim)

    # print(us_dec, us_inc)
    for idx, (u_inc, u_dec) in enumerate(zip(us_inc, us_dec)):
        # calculate partial differential w.r.t. u
        state_inc = simulate_dynamics(env, x.copy(), u_inc, dt)
        state_dec = simulate_dynamics(env, x.copy(), u_dec, dt)
        B[:, idx] = (state_inc - state_dec) / (2 * delta)

    return B


u = None
def calc_lqr_input(env, sim_env, debug_flag=False):
    """Calculate the optimal control input for the given state.

    If you are following the API and simulate dynamics is returning
    xdot, then you should use the scipy.linalg.solve_continuous_are
    function to solve the Ricatti equations.

    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.

    Returns
    -------
    u: np.array
      The command to execute at this point.
    """

    # prepare whatever need later
    global u
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    x = env.state.copy()
    goal_q = env.goal_q.copy()
    goal_dq = env.goal_dq.copy()
    Q = env.Q.copy()
    R = env.R.copy()

    if u is None:
      u = np.zeros(action_dim)

    # calcuate A and B matrix
    A = approximate_A(sim_env, x.copy(), u.copy())
    if debug_flag : prGreen('A={}'.format(A))
    assert(A.shape == (state_dim, state_dim))

    B = approximate_B(sim_env, x.copy(), u.copy())
    if debug_flag : prYellow('B={}'.format(B))
    assert(B.shape == (state_dim, action_dim))

    # solve CARE, return zero if raise error
    try:
      X = scipy.linalg.solve_continuous_are(A, B, Q, R)
    except:
      return np.zeros(action_dim)
    
    # calculate K gain
    K = np.dot(np.linalg.pinv(R), np.dot(B.T, X))
    if debug_flag : prRed('K={}'.format(K))

    # calcuate action
    x_target = x[:2]-goal_q
    u = np.hstack((x_target, x[2:]))
    u = -np.dot(K, u)
    u = np.clip(u, 
      env.action_space.low, env.action_space.high)
    if debug_flag : prGreen((x[:2], goal_q))

    return u
