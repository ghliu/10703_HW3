"""LQR, iLQR and MPC."""

from deeprl_hw3.controllers import approximate_A, approximate_B
import numpy as np
import scipy.linalg
import time

from ipdb import set_trace as debug

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

    env.state = x.copy()
    x1, _, _, _ = env.step(u)
    return x1


def cost_inter(env, x, u, discrete=False):
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
    discrete: boolean
      if True, multiply the env.dt

    Returns
    -------
    l, l_x, l_xx, l_u, l_uu, l_ux. The first term is the loss, where the remaining terms are derivatives respect to the
    env, corresponding variables, ex: (1) l_x is the first order derivative d l/d x (2) l_xx is the second order derivative
    d^2 l/d x^2
    """
    num_actions = env.action_space.shape[0]
    num_states = env.observation_space.shape[0]
    dt = env.dt if discrete else 1.

    l = np.sum(u**2)*dt
    l_x = np.zeros(num_states)*dt
    l_xx = np.zeros((num_states, num_states))*dt
    l_u = 2 * u*dt
    l_uu = 2 * np.eye(num_actions)*dt
    l_ux = np.zeros((num_actions, num_states))*dt

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

    num_states = env.observation_space.shape[0]
    target = env.goal.copy()

    # calculate l, l_x, l_xx
    weight = 1e4
    l = weight * np.sum((x-target)**2)
    l_x = 2.*weight*(x-target)
    l_xx = 2.*weight*np.eye(num_states)
    
    return l, l_x, l_xx

    
def simulate(env, x0, U):
    tN = U.shape[0]
    num_states = env.observation_space.shape[0]

    X = np.zeros((tN, num_states))
    X[0] = x0
    cost = 0.
    for t in range(tN-1):
        X[t+1] = simulate_dynamics_next(env, X[t], U[t])
        l,_,_,_,_,_ = cost_inter(env, X[t], U[t], discrete=True)
        cost += l

    l_f,_,_ = cost_final(env, X[-1])
    cost += l_f

    return X, cost


def solve(env, x0, U, max_iter, useLM, debug_flag):
    # initialize paramters
    action_dim = env.action_space.shape[0] 
    state_dim = env.observation_space.shape[0]

    sim_new_trajectory = True 
    lamb = 1.0
    lamb_factor = 10
    lamb_max = 1000
    eps_converge = 0.001
    tN = U.shape[0]

    x0 = env.state.copy()
    for ii in range(int(max_iter)):
        # f, c            
        if sim_new_trajectory: 

            X, cost = simulate(env, x0, U)
            oldcost = np.copy(cost)

            # linearized dynamics
            f_x = np.zeros((tN, state_dim, state_dim)) 
            f_u = np.zeros((tN, state_dim, action_dim)) 

            # quadratized cost function 
            l = np.zeros((tN,1)) 
            l_x = np.zeros((tN, state_dim))
            l_xx = np.zeros((tN, state_dim, state_dim))
            l_u = np.zeros((tN, action_dim)) 
            l_uu = np.zeros((tN, action_dim, action_dim)) 
            l_ux = np.zeros((tN, action_dim, state_dim))

            for t in range(tN-1):
                A = approximate_A(env, X[t].copy(), U[t].copy(), dt=env.dt)
                B = approximate_B(env, X[t].copy(), U[t].copy(), dt=env.dt) 
                f_x[t] = np.eye(state_dim) + A * env.dt
                f_u[t] = B * env.dt
                
                (l[t], l_x[t], l_xx[t], l_u[t], l_uu[t], l_ux[t]) = \
                    cost_inter(env, X[t], U[t], discrete=True)

            l[-1], l_x[-1], l_xx[-1] = cost_final(env, X[-1])   
            sim_new_trajectory = False

        # f, k update
        V = l[-1].copy() # value function
        V_x = l_x[-1].copy() # dV / dx
        V_xx = l_xx[-1].copy() # d^2 V / dx^2
        k = np.zeros((tN, action_dim)) # feedforward modification
        K = np.zeros((tN, action_dim, state_dim)) # feedback gain 

        # backward
        for t in range(tN-2, -1, -1):
            Q_x = l_x[t] + np.dot(f_x[t].T, V_x) 
            Q_u = l_u[t] + np.dot(f_u[t].T, V_x)

            Q_xx = l_xx[t] + np.dot(f_x[t].T, np.dot(V_xx, f_x[t])) 
            Q_ux = l_ux[t] + np.dot(f_u[t].T, np.dot(V_xx, f_x[t]))
            Q_uu = l_uu[t] + np.dot(f_u[t].T, np.dot(V_xx, f_u[t]))

            # Calculate Q_uu^-1 with regularization term set by 
            # Levenberg-Marquardt heuristic (at end of this loop)
            Q_uu_evals, Q_uu_evecs = np.linalg.eig(Q_uu)
            Q_uu_evals[Q_uu_evals < 0] = 0.0
            Q_uu_evals += lamb
            Q_uu_inv = np.dot(Q_uu_evecs, 
                np.dot(np.diag(1.0/Q_uu_evals), Q_uu_evecs.T))

            k[t] = -np.dot(Q_uu_inv, Q_u)
            K[t] = -np.dot(Q_uu_inv, Q_ux)

            V_x = Q_x - np.dot(K[t].T, np.dot(Q_uu, k[t]))
            V_xx = Q_xx - np.dot(K[t].T, np.dot(Q_uu, K[t]))


        # forward
        Unew = np.zeros((tN, action_dim))
        xnew = x0.copy() 
        for t in range(tN - 1): 
            Unew[t] = U[t] + k[t] + np.dot(K[t], xnew - X[t]) 
            xnew = simulate_dynamics_next(env, xnew, Unew[t]) 

        # evaluate the new trajectory 
        Xnew, costnew = simulate(env, x0, Unew)

        # Levenberg-Marquardt heuristic
        if costnew <= cost: 

            if useLM:
                lamb /= lamb_factor

            X = np.copy(Xnew) # update trajectory 
            U = np.copy(Unew) # update control signal
            oldcost = np.copy(cost)
            cost = np.copy(costnew)

            sim_new_trajectory = True 

            if ii > 0 and ((abs(oldcost-cost)/cost) < eps_converge):
                if debug_flag: 
                    print("Converged at iteration = %d; Cost = %.4f;"%(ii,costnew) + 
                            " logLambda = %.1f"%np.log(lamb))
                break

        else: 
            #  gradient descent
            if useLM:
                lamb *= lamb_factor
            if debug_flag and lamb > lamb_max: 
                print("lambda > max_lambda at iteration = %d;"%ii + 
                        " Cost = %.4f; logLambda = %.1f"%(cost, 
                                                          np.log(lamb)))

    return X, U, cost


U = None
t = -1
def calc_ilqr_input(env, sim_env, 
    tN=50, max_iter=1e6, useLM=False, debug_flag=False):
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

    action_dim = env.action_space.shape[0]

    # update t
    t = np.mod(t+1,tN)
    sim_env.state = env.state.copy()

    if U is None:
        if debug_flag: t0 = time.time()
        U = np.zeros((tN,action_dim))
        x0 = sim_env.state.copy()
        X, U, cost = solve(sim_env, x0, U, max_iter, useLM, debug_flag)
        if debug_flag: print('solve with {}'.format(time.time()-t0))

    u = U[t]

    # reset
    if t == tN-1:
        if debug_flag: print('reset')
        U = None

    return u
