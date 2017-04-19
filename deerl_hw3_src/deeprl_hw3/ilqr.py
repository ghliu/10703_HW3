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

    xy = x.copy() # copy? 
    target = env.goal_q.copy()

    xy_err = np.array([xy[0] - target[0], xy[1] - target[1]])
    l = (wp * np.sum(xy_err**2) + wv * np.sum(x[num_actions:num_actions*2]**2))

    l_x[:num_actions] = wp * diff(x[:num_actions], env)
    l_x[num_actions:num_actions*2] = (2 * wv * x[num_actions:num_actions*2])

    eps = 1e-4 

    for k in range(num_actions): 
        veps = np.zeros(num_actions)
        veps[k] = eps
        d1 = wp * diff(x[0:num_actions] + veps, env)
        d2 = wp * diff(x[0:num_actions] - veps, env)
        l_xx[0:num_actions, k] = ((d1-d2) / 2.0 / eps).flatten()

    l_xx[num_actions:num_actions*2, num_actions:num_actions*2] = 2 * wv * np.eye(num_actions)
    return l, l_x, l_xx


def diff(x, env):
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
        X[t+1] = simulate_dynamics_next(env, X[t], U[t])
        l,_,_,_,_,_ = cost_inter(env, X[t], U[t])
        cost = cost + dt * l

    l_f,_,_ = cost_final(env, X[-1])
    cost = cost + l_f

    return X, cost

U = None
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
    global U
    action_dim = env.action_space.shape[0] 
    state_dim = env.observation_space.shape[0]
    if U == None:
        U = np.zeros((tN, action_dim))

    # initialize paramters: 
    sim_new_trajectory = True 
    lamb = 1.0
    lamb_factor = 10
    lamb_max = 1000
    eps_converge = 0.001

    x0 = env.state.copy()
    for ii in range(int(max_iter)):
        # f, c            
        if sim_new_trajectory: 

            X, cost = simulate(env, x0, U)
            oldcost = np.copy(cost) # copy for exit condition check

            # for storing linearized dynamics
            # x(t+1) = f(x(t), u(t))
            f_x = np.zeros((tN, state_dim, state_dim)) # df / dx
            f_u = np.zeros((tN, state_dim, action_dim)) # df / du

            # for storing quadratized cost function 
            l = np.zeros((tN,1)) # immediate state cost 
            l_x = np.zeros((tN, state_dim)) # dl / dx
            l_xx = np.zeros((tN, state_dim, state_dim)) # d^2 l / dx^2
            l_u = np.zeros((tN, action_dim)) # dl / du
            l_uu = np.zeros((tN, action_dim, action_dim)) # d^2 l / du^2
            l_ux = np.zeros((tN, action_dim, state_dim)) # d^2 l / du / dx

            for t in range(tN-1):
                # x(t+1) = f(x(t), u(t)) = x(t) + dx(t) * dt
                # linearized dx(t) = np.dot(A(t), x(t)) + np.dot(B(t), u(t))
                # f_x = np.eye + A(t)
                # f_u = B(t)

                A = approximate_A(sim_env, X[t].copy(), U[t].copy())
                B = approximate_B(sim_env, X[t].copy(), U[t].copy()) 
                f_x[t] = np.eye(state_dim) + A * env.dt
                f_u[t] = B * env.dt
                
                (l[t], l_x[t], l_xx[t], l_u[t], l_uu[t], l_ux[t]) = cost_inter(env, X[t], U[t])
                l[t] *= env.dt
                l_x[t] *= env.dt
                l_xx[t] *= env.dt
                l_u[t] *= env.dt
                l_uu[t] *= env.dt
                l_ux[t] *= env.dt

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
            # 4a) Q_x = l_x + np.dot(f_x^T, V'_x)
            Q_x = l_x[t] + np.dot(f_x[t].T, V_x) 
            # 4b) Q_u = l_u + np.dot(f_u^T, V'_x)
            Q_u = l_u[t] + np.dot(f_u[t].T, V_x)

            # 4c) Q_xx = l_xx + np.dot(f_x^T, np.dot(V'_xx, f_x)) + np.einsum(V'_x, f_xx)
            Q_xx = l_xx[t] + np.dot(f_x[t].T, np.dot(V_xx, f_x[t])) 
            # 4d) Q_ux = l_ux + np.dot(f_u^T, np.dot(V'_xx, f_x)) + np.einsum(V'_x, f_ux)
            Q_ux = l_ux[t] + np.dot(f_u[t].T, np.dot(V_xx, f_x[t]))
            # 4e) Q_uu = l_uu + np.dot(f_u^T, np.dot(V'_xx, f_u)) + np.einsum(V'_x, f_uu)
            Q_uu = l_uu[t] + np.dot(f_u[t].T, np.dot(V_xx, f_u[t]))

            # Calculate Q_uu^-1 with regularization term set by 
            # Levenberg-Marquardt heuristic (at end of this loop)
            Q_uu_evals, Q_uu_evecs = np.linalg.eig(Q_uu)
            Q_uu_evals[Q_uu_evals < 0] = 0.0
            Q_uu_evals += lamb
            Q_uu_inv = np.dot(Q_uu_evecs, np.dot(np.diag(1.0/Q_uu_evals), Q_uu_evecs.T))

            # 5b) k = -np.dot(Q_uu^-1, Q_u)
            k[t] = -np.dot(Q_uu_inv, Q_u)
            # 5b) K = -np.dot(Q_uu^-1, Q_ux)
            K[t] = -np.dot(Q_uu_inv, Q_ux)

            # 6a) DV = -.5 np.dot(k^T, np.dot(Q_uu, k))
            # 6b) V_x = Q_x - np.dot(K^T, np.dot(Q_uu, k))
            V_x = Q_x - np.dot(K[t].T, np.dot(Q_uu, k[t]))
            # 6c) V_xx = Q_xx - np.dot(-K^T, np.dot(Q_uu, K))
            V_xx = Q_xx - np.dot(K[t].T, np.dot(Q_uu, K[t]))


        # forward
        Unew = np.zeros((tN, action_dim))
        # calculate the optimal change to the control trajectory
        xnew = x0.copy() # 7a)
        for t in range(tN - 1): 
            Unew[t] = U[t] + k[t] + np.dot(K[t], xnew - X[t]) # 7b)
            xnew = simulate_dynamics_next(env, xnew, Unew[t]) # 7c)

        # evaluate the new trajectory 
        Xnew, costnew = simulate(env, x0, Unew)

        # Levenberg-Marquardt heuristic
        if costnew < cost: 
            # decrease lambda (get closer to Newton's method)
            lamb /= lamb_factor

            X = np.copy(Xnew) # update trajectory 
            U = np.copy(Unew) # update control signal
            oldcost = np.copy(cost)
            cost = np.copy(costnew)

            sim_new_trajectory = True # do another rollout

            if ii > 0 and ((abs(oldcost-cost)/cost) < eps_converge):
                print("Converged at iteration = %d; Cost = %.4f;"%(ii,costnew) + 
                            " logLambda = %.1f"%np.log(lamb))
                break

        else: 
            # increase lambda (get closer to gradient descent)
            lamb *= lamb_factor
            # print("cost: %.4f, increasing lambda to %.4f")%(cost, lamb)
            if lamb > lamb_max: 
                print("lambda > max_lambda at iteration = %d;"%ii + 
                        " Cost = %.4f; logLambda = %.1f"%(cost, 
                                                          np.log(lamb)))
                break
        # update x, u

    return U #np.zeros((50, 2))
