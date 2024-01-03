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
        return coeffs.T @ cost_mat_full @ coeffs + jnp.exp(regularizer[0].apply(regularizer[1], coeffs)[0])

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