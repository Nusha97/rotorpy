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
        return coeffs.T @ cost_mat_full @ coeffs + regularizer(coeffs)[0]

    pg = ProjectedGradient(
        nn_cost,
        projection=projection_affine_set,
        maxiter=5,
        # verbose=True,
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