import numpy as np
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_affine_set
import jax.numpy as jnp
import jax


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
        # print(regularizer(coeffs))
        return coeffs.T @ cost_mat_full @ coeffs + regularizer.apply_fn(regularizer.params, coeffs)[0]
    
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


    pg = ProjectedGradient(
        nn_cost,
        projection=projection_affine_set,
        maxiter=5, # 1, if nan, return initial
        verbose=True,
    )
    # sol = pg.run(coeff0)
    # pg = ProjectedGradient(nn_cost, projection=projection_affine_set)
    # print("A_coeff_full", A_coeff_full)
    # print("b_coeff_full", b_coeff_full)
    # print("cost_mat_full", cost_mat_full)
    # grad_fn = jax.value_and_grad()
    # _, grads = grad_fn(init_ref, aug_test_state[i, :][:, None])
    # new_ref = init_ref - 0.001 * onp.ravel(grads)
    sol = pg.run(coeff0, hyperparams_proj=(A_coeff_full, b_coeff_full))
    if np.isnan(sol.state.error):
        return None, None  # Return None if NaN encountered
    else:
        coeff = sol.params
        pred = sol.state.error

        print("Norm difference", np.linalg.norm(coeff0.ravel() - coeff))

        return coeff, pred



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