from functools import partial
import numpy as np
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_affine_set
import jax.numpy as jnp
import jax
from flax.linen import jit
from flax.linen import custom_vjp
# from jax import jit



# @jit
def modify_reference(
    regularizer,
    cost_mat_full,
    A_coeff_full,
    b_coeff_full,
    coeff0,
    # maxiter=5
):
    """
    Running projected gradient descent on the neural network cost + min snap cost with constraints
    """
    # @jit
    # @jax.jit
    def nn_cost(coeffs):
        """
        Function to compute trajectories given polynomial coefficients
        :param coeffs: 4-dim polynomial coefficients (x, y, z, yaw)
        :param ts: waypoint time allocation
        :param numsteps: Total number of samples in the reference
        :return: ref
        """
        # we use log(cost) during training, so we need to exponentiate it here
        # return coeffs.T @ cost_mat_full @ coeffs + jnp.exp(regularizer[0].apply(regularizer[1], coeffs)[0])
        # return coeffs.T @ cost_mat_full @ coeffs + regularizer.apply_fn(regularizer.params, coeffs)[0]
        # print(regularizer(coeffs))
        # return coeffs.T @ cost_mat_full @ coeffs + regularizer(coeffs)[0]
        # return coeffs.T @ cost_mat_full @ coeffs + regularizer[0].apply(regularizer[1], coeffs)[0]
        return coeffs.T @ cost_mat_full @ coeffs + jnp.exp(regularizer[0].apply(regularizer[1], coeffs)[0])
        
    
    # def run_optimization(coeff_init):
    #     pg = ProjectedGradient(
    #         nn_cost,
    #         projection=projection_affine_set,
    #         maxiter=1,
    #         verbose=True,
    #     )
    #     sol = pg.run(coeff_init, hyperparams_proj=(A_coeff_full, b_coeff_full))
    #     return sol.params, sol.state.error

    # current_coeffs = coeff0
    # print("Initial coefficients:", coeff0)
    # for iter_num in range(maxiter):
    #     new_coeffs, error = run_optimization(current_coeffs)
    #     print(f"Iteration {iter_num}: Coefficients:", new_coeffs)

    #     if np.isnan(error):
    #         print(f"NaN encountered in iteration {iter_num}.")
    #         if iter_num == 0:
    #             # First iteration, return initial coefficients
    #             return coeff0, error
    #         else:
    #             # Not first iteration, return last valid coefficients
    #             return current_coeffs, error
    #     current_coeffs = new_coeffs  # Update coefficients for next iteration

    # print("Optimization completed without NaN errors.")
    # print("Final coefficients:", current_coeffs)
    # return current_coeffs, error
    
    # Initialize ProjectedGradient with maxiter set to 1
    pg = ProjectedGradient(
        nn_cost,
        projection=projection_affine_set,
        maxiter=1,
        # jit = True,
        verbose=True,
    )

    # Run the initial step of ProjectedGradient
    sol = pg.run(coeff0, hyperparams_proj=(A_coeff_full, b_coeff_full))

    # Initialize variables to track the best solution and error
    best_solution = sol.params
    best_error = sol.state.error
    nan_encountered = np.isnan(best_error)

    # If NaN error is encountered at the beginning, return immediately
    if nan_encountered:
        print("Final lowest ProximalGradient error: NaN")
        return coeff0, best_error, nan_encountered

    # Total iterations, adjust this number as needed
    total_iterations = 30

    # Iteratively update and check for the best solution
    for _ in range(total_iterations - 1):
        sol = pg.update(sol.params, sol.state, hyperparams_proj=(A_coeff_full, b_coeff_full))

        # Check for NaN errors
        current_error_nan = np.isnan(sol.state.error)
        if current_error_nan:
            nan_encountered = True
            continue

        # Update best solution if the current solution has a lower error
        if sol.state.error < best_error:
            best_solution = sol.params
            best_error = sol.state.error
            print(f"New lowest ProximalGradient error: {best_error}")

    print(f"Final lowest ProximalGradient error: {best_error}")
    return best_solution, best_error, nan_encountered
    """
    pg = ProjectedGradient(
        nn_cost,
        projection=projection_affine_set,
        # maxiter=9, # 1, if nan, return initial
        maxiter=30,
        # stepsize=-1e-5,
        # maxls=10,
        # jit = True,
        verbose=True,
        # tol=10
        # implicit_diff = False,
        # implicit_differentiation=True,  # Enable implicit differentiation
    )
    sol = pg.run(coeff0, hyperparams_proj=(A_coeff_full, b_coeff_full))

    nan_encountered = np.isnan(sol.state.error)
    print("nan_encountered", nan_encountered)
    return sol.params, sol.state.error, nan_encountered
    """
    



def main():
    # Test code here
    def regularizer(x):
        return jnp.sum(x ** 2)

    A = 4 * jnp.eye(2)
    b = 2.0 * jnp.ones(2)

    H = 10.0 * jnp.eye(2)
    coeff, pred = modify_reference(regularizer, H, A, b, jnp.array([1.0, 0]))
    print(coeff)
    print(pred)


if __name__ == '__main__':
    main()