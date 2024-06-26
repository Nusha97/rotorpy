import jax.numpy as jnp
import numpy as np
import jax.scipy.linalg as spl
from .trajutils import _cost_matrix
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_affine_set
from trajgen.nonlinear import _coeff_constr_A, _coeff_constr_b


def modify_reference(wp, ts, numsteps, order, p, regularizer, coeff0):
    """
    Running projected gradient descent on the neural network cost + min snap cost with constraints
    """
    num_seg = len(ts) - 1
    n = coeff0.shape[2]
    num_coeffs = np.prod(coeff0.shape[1:])
    durations = ts[1:] - ts[:-1]
    cost_mat = spl.block_diag(*[_cost_matrix(order, num_seg, d) for d in durations])
    A_coeff = _coeff_constr_A(ts, n, num_coeffs)
    b_coeff = _coeff_constr_b(wp.T, ts, n)

    cost_mat_full = spl.block_diag(*[cost_mat for i in range(p)])
    A_coeff_full = spl.block_diag(*[A_coeff for i in range(p)])
    b_coeff_full = jnp.ravel(b_coeff)
    times = np.concatenate([np.linspace(0, ts[i+1]-ts[i], numsteps // num_seg) for i in range(num_seg)])
    print("Times", times.shape)

    #@jit
    def nn_cost(coeffs):
        """
        Function to compute trajectories given polynomial coefficients
        :param coeffs: 4-dim polynomial coefficients (x, y, z, yaw)
        :param ts: waypoint time allocation
        :param numsteps: Total number of samples in the reference
        :return: ref
        """
        ref = jnp.zeros((p, numsteps))

        for j in range(p):
                # ref.at[:, i].set(jnp.dot(_diff_coeff(coeffs.shape[2] - 1, tt - ts[k], 0), coeffs[:, k, :].T))
            ref = ref.at[j, :].set(jnp.polyval(coeffs[j * num_seg * (order + 1) : (j + 1) * num_seg * (order + 1)], times))
                # ref.append(jnp.polyval(coeffs[j, :], tt - ts[k]))
            # ref.append(poly[k](tt - ts[k]))
        # print("Network cost", jnp.exp(regularizer(jnp.append(wp[0, :], jnp.vstack(ref)))[0]))
        # print("Coeff cost", (coeffs.T @ cost_mat_full @ coeffs))
        return coeffs.T @ cost_mat_full @ coeffs + jnp.exp(regularizer(jnp.append(wp[0, :], jnp.vstack(ref)))[0])
        #for j in range(p):
        #    ref = ref.at[j, :].set(jnp.polyval(coeffs[j * num_seg * (order + 1):(j + 1) * num_seg * (order + 1)], times))
        #return coeffs.T @ cost_mat_full @ coeffs + regularizer(jnp.append(wp[0, :], jnp.vstack(ref)))[0]


    pg = ProjectedGradient(nn_cost, projection=projection_affine_set)
    sol = pg.run(coeff0.ravel(), hyperparams_proj=(A_coeff_full, b_coeff_full))
    coeff = sol.params
    pred = sol.state.error

    print("Norm difference", np.linalg.norm(coeff0.ravel() - coeff))

    return np.reshape(coeff, (p, num_seg, order+1)), pred
