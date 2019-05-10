import cvxpy as cvx
import pandas as pd 
import numpy as np 

def NMF(A, k, constant=0.01, regularisation=False, MAX_ITERS=100, cost='absolute'):
    """

    Parameters
    ----------
    A: matrix to be decomposed (m rows and n columns)
    k: number of latent factors
    constant: coefficient of regularisation
    regularisation: whether to use regularisation
    MAX_ITERS: maximium number of iterations
    cost: 'absolute' or 'relative'

    Returns
    -------
    W:
    H:
    Residual

    """

    np.random.seed(0)

    m, n = A.shape
    mask = ~np.isnan(A)
    
    # Initialize W randomly.
    W_init = np.random.rand(m, k)
    W = W_init

    # Perform alternating minimization.

    residual = np.zeros(MAX_ITERS)
    for iter_num in range(1, 1 + MAX_ITERS):
        # For odd iterations, treat W constant, optimize over H.
        if iter_num % 2 == 1:
            H = cvx.Variable(k, n)
            constraint = [H >= 0]
            
        # For even iterations, treat X constant, optimize over Y.
        else:
            W = cvx.Variable(m, k)

            constraint = [W >= 0]
           
        Temp = W*H
        
        one_A = cvx.Constant(1.0 / (A[mask]+1e-3))
        abs_error = A[mask] - (W * H)[mask]
        rel_error = cvx.mul_elemwise(one_A, abs_error)
        if cost=='absolute':
            error=abs_error
        else:
            # If relative cost, 
            error = rel_error
            
        # Solve the problem.
        if not regularisation:
            obj = cvx.Minimize(cvx.norm(error, 'fro'))

        else:
            if iter_num % 2 == 1:
                obj = cvx.Minimize(cvx.norm(error, 'fro') + constant * cvx.norm(H))
            else:
                obj = cvx.Minimize(cvx.norm(error, 'fro') + constant * cvx.norm(W))

        prob = cvx.Problem(obj, constraint)
        prob.solve(solver=cvx.SCS)

        if prob.status != cvx.OPTIMAL:
            pass
       
        residual[iter_num - 1] = prob.value
        # Convert variable to NumPy array constant for next iteration.
        if iter_num % 2 == 1:
            H = H.value
        else:
            W = W.value
    return W, H, residual