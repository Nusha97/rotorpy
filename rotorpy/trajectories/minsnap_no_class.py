import numpy as np
import cvxopt
from scipy.linalg import block_diag
import itertools
from learning.trajgen import nonlinear_jax, nonlinear, sgd_jax
import torch


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    """
    From https://scaron.info/blog/quadratic-programming-in-python.html . Infrastructure code for solving quadratic programs using CVXOPT.
    The structure of the program is as follows:

    min 0.5 xT P x + qT x
    s.t. Gx <= h
         Ax = b
    Inputs:
        P, numpy array, the quadratic term of the cost function
        q, numpy array, the linear term of the cost function
        G, numpy array, inequality constraint matrix
        h, numpy array, inequality constraint vector
        A, numpy array, equality constraint matrix
        b, numpy array, equality constraint vector
    Outputs:
        The optimal solution to the quadratic program
    """
    P = 0.5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
    if A is not None:
        args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    # show progress false
    cvxopt.solvers.options["show_progress"] = False
    sol = cvxopt.solvers.qp(*args)
    if "optimal" not in sol["status"]:
        return None
    return np.array(sol["x"]).reshape((P.shape[1],))


def H_fun(dt, k=7):
    """
    Computes the cost matrix for a single segment in a single dimension.
    *** Assumes that the decision variables c_i are e.g. x(t) = c_0 + c_1*t + c_2*t^2 + c_3*t^3 + c_4*t^4 + c_5*t^5 + .. + c_k*t^k
    Inputs:
        dt, scalar, the duration of the segment (t_(i+1) - t_i)
    Outputs:
        H, numpy array, matrix containing the min snap cost function for that segment. Assumes the polynomial is at least order 5.
    """

    H = np.zeros((k + 1, k + 1))

    seventh_order_cost = np.array(
        [
            [576 * dt, 1440 * dt**2, 2880 * dt**3, 5040 * dt**4],
            [1440 * dt**2, 4800 * dt**3, 10800 * dt**4, 20160 * dt**5],
            [2880 * dt**3, 10800 * dt**4, 25920 * dt**5, 50400 * dt**6],
            [5040 * dt**4, 20160 * dt**5, 50400 * dt**6, 100800 * dt**7],
        ]
    )

    # Only take up to the (k+1) entries
    cost = seventh_order_cost[0 : (k + 1 - 4), 0 : (k + 1 - 4)]

    H[4 : (k + 1), 4 : (k + 1)] = cost

    return H


def get_1d_constraints(keyframes, delta_t, m, k=7, vmax=5, vstart=0, vend=0):
    """
    Computes the constraint matrices for the min snap problem.
    *** Assumes that the decision variables c_i are e.g. o(t) = c_0 + c_1*t + c_2*t^2 + ... c_(k)*t^(k)

    We impose the following constraints FOR EACH SEGMENT m:
        1) x_m(0) = keyframe[i]             # position at t = 0
        2) x_m(dt) = keyframe[i+1]          # position at t = dt
        3) v_m(dt) = v_(m+1)(0)             # velocity continuity for interior segments
        4) a_m(dt) = a_(m+1)(0)             # acceleration continuity for interior segments
        5) j_m(dt) = j_(m+1)(0)             # jerk continuity for interior segments
        6) s_m(dt) = s_(m+1)(0)             # snap continuity for interior segments

    Inputs:
        keyframes, numpy array, a list of m waypoints IN ONE DIMENSION (x,y,z, or yaw)
        delta_t, numpy array, the times between keyframes computed apriori.
        m, int, the number of segments.
        k, int, the degree of the polynomial.
        vmax, float, max speeds imposed at the midpoint of each segment.
        vstart, float, the starting speed of the quadrotor.
        vend, float, the ending speed of the quadrotor.
    Outputs:
        A, numpy array, matrix of equality constraints (left side).
        b, numpy array, array of equality constraints (right side).
        G, numpy array, matrix of inequality constraints (left side).
        h, numpy array, array of inequality constraints (right side).

    """

    # The constraint matrices to be filled out.
    A = []
    b = []
    G = []
    h = []

    for i in range(m):  # for each segment...
        # Gets the segment duration
        dt = delta_t[i]

        # Position continuity at the beginning of the segment
        A.append([0] * (k + 1) * i + [1] + [0] * (k) + [0] * (k + 1) * (m - i - 1))
        b.append(keyframes[i])

        # Position continuity at the end of the segment
        A.append(
            [0] * (k + 1) * i
            + [dt**j for j in range(k + 1)]
            + [0] * (k + 1) * (m - i - 1)
        )
        b.append(keyframes[i + 1])

        # Intermediate smoothness constraints
        if i < (m - 1):  # we don't want to include the last segment for this loop
            A.append(
                [0] * (k + 1) * i
                + [0]
                + [-j * dt ** (j - 1) for j in range(1, k + 1)]
                + [0]
                + [j * (0) ** (j - 1) for j in range(1, k + 1)]
                + [0] * (k + 1) * (m - i - 2)
            )  # Velocity
            b.append(0)
            A.append(
                [0] * (k + 1) * i
                + [0] * 2
                + [-(j - 1) * j * dt ** (j - 2) for j in range(2, k + 1)]
                + [0] * 2
                + [(j - 1) * j * (0) ** (j - 2) for j in range(2, k + 1)]
                + [0] * (k + 1) * (m - i - 2)
            )  # Acceleration
            b.append(0)
            A.append(
                [0] * (k + 1) * i
                + [0] * 3
                + [-(j - 2) * (j - 1) * j * dt ** (j - 3) for j in range(3, k + 1)]
                + [0] * 3
                + [(j - 2) * (j - 1) * j * (0) ** (j - 3) for j in range(3, k + 1)]
                + [0] * (k + 1) * (m - i - 2)
            )  # Jerk
            b.append(0)
            A.append(
                [0] * (k + 1) * i
                + [0] * 4
                + [
                    -(j - 3) * (j - 2) * (j - 1) * j * dt ** (j - 4)
                    for j in range(4, k + 1)
                ]
                + [0] * 4
                + [
                    (j - 3) * (j - 2) * (j - 1) * j * (0) ** (j - 4)
                    for j in range(4, k + 1)
                ]
                + [0] * (k + 1) * (m - i - 2)
            )  # Snap
            b.append(0)

        # Inequality constraints
        G.append(
            [0] * (k + 1) * i
            + [0]
            + [j * (0.5 * dt) ** (j - 1) for j in range(1, k + 1)]
            + [0] * (k + 1) * (m - i - 1)
        )  # Velocity constraint at midpoint
        h.append(vmax)

    A.append(
        [0] + [j * (0) ** (j - 1) for j in range(1, k + 1)] + [0] * (k + 1) * (m - 1)
    )  # Velocity at start
    b.append(vstart)
    A.append(
        [0] * (k + 1) * (m - 1) + [0] + [j * (dt) ** (j - 1) for j in range(1, k + 1)]
    )  # Velocity at end
    b.append(vend)
    A.append(
        [0] * 2
        + [(j - 1) * j * (0) ** (j - 2) for j in range(2, k + 1)]
        + [0] * (k + 1) * (m - 1)
    )  # Acceleration = 0 at start
    b.append(0)
    A.append(
        [0] * (k + 1) * (m - 1)
        + [0] * 2
        + [(j - 1) * j * (dt) ** (j - 2) for j in range(2, k + 1)]
    )  # Acceleration = 0 at end
    b.append(0)
    A.append(
        [0] * 3
        + [(j - 2) * (j - 1) * j * (0) ** (j - 3) for j in range(3, k + 1)]
        + [0] * (k + 1) * (m - 1)
    )  # Jerk = 0 at start
    b.append(0)
    A.append(
        [0] * (k + 1) * (m - 1)
        + [0] * 3
        + [(j - 2) * (j - 1) * j * (dt) ** (j - 3) for j in range(3, k + 1)]
    )  # Jerk = 0 at end
    b.append(0)

    # Convert to numpy arrays and ensure floats to work with cvxopt.
    A = np.array(A).astype(float)
    b = np.array(b).astype(float)
    G = np.array(G).astype(float)
    h = np.array(h).astype(float)

    return (A, b, G, h)

# similar to Minsnap class's init function
def compute_min_snap_coefficients(points, yaw_angles, poly_degree, yaw_poly_degree, v_max, v_avg, v_start, v_end, yaw_rate_max, use_neural_network=False, regularizer=None):
    # Check if yaw_angles are provided
    yaw = np.zeros((points.shape[0])) if yaw_angles is None else yaw_angles
    
    # Compute the distances between each waypoint
    seg_dist = np.linalg.norm(np.diff(points, axis=0), axis=1)
    seg_mask = np.append(True, seg_dist > 1e-1)
    filtered_points = points[seg_mask, :]
    
    m = filtered_points.shape[0] - 1  # Number of segments

    # Handle case with fewer than two waypoints
    if m < 1:
        # Return appropriate values for this case
        return {
            'x_poly': np.zeros((1, 3, 6)),
            'x_poly': np.zeros((1, 3, 6)),
            # Add other necessary outputs with default or zero values
        }

    # Compute segment durations and keyframe times
    delta_t = seg_dist / v_avg
    t_keyframes = np.concatenate(([0], np.cumsum(delta_t)))

    # Construct cost matrices for position and yaw
    H_pos = [H_fun(delta_t[i], k=poly_degree) for i in range(m)]
    H_yaw = [H_fun(delta_t[i], k=yaw_poly_degree) for i in range(m)]
    P_pos = block_diag(*H_pos)
    P_yaw = block_diag(*H_yaw)
    q_pos = np.zeros(((poly_degree + 1) * m, 1))
    q_yaw = np.zeros(((yaw_poly_degree + 1) * m, 1))

    # Construct constraints for each dimension
    (Ax, bx, Gx, hx) = get_1d_constraints(filtered_points[:, 0], delta_t, m, poly_degree, v_max, v_start[0], v_end[0])
    (Ay, by, Gy, hy) = get_1d_constraints(filtered_points[:, 1], delta_t, m, poly_degree, v_max, v_start[1], v_end[1])
    (Az, bz, Gz, hz) = get_1d_constraints(filtered_points[:, 2], delta_t, m, poly_degree, v_max, v_start[2], v_end[2])
    (Ayaw, byaw, Gyaw, hyaw) = get_1d_constraints(yaw, delta_t, m, yaw_poly_degree, yaw_rate_max)

    # Solve for optimal coefficients
    c_opt_x = cvxopt_solve_qp(P_pos, q=q_pos, G=Gx, h=hx, A=Ax, b=bx)
    c_opt_y = cvxopt_solve_qp(P_pos, q=q_pos, G=Gy, h=hy, A=Ay, b=by)
    c_opt_z = cvxopt_solve_qp(P_pos, q=q_pos, G=Gz, h=hz, A=Az, b=bz)
    c_opt_yaw = cvxopt_solve_qp(P_yaw, q=q_yaw, G=Gyaw, h=hyaw, A=Ayaw, b=byaw)

    # Check for solution validity
    if c_opt_x is None or c_opt_y is None or c_opt_z is None or c_opt_yaw is None:
        # Handle the case where the QP solver fails to find a solution
        return None  # Or an appropriate error handling

    # Neural Network integration
    if use_neural_network:
        # Prepare data for the neural network
        min_snap_coeffs = np.concatenate([c_opt_x, c_opt_y, c_opt_z, c_opt_yaw])
        H = block_diag(
            *[0.5 * (P_pos.T + P_pos), 0.5 * (P_pos.T + P_pos), 0.5 * (P_pos.T + P_pos), 0.5 * (P_yaw.T + P_yaw)]
        )  # Cost function is the same for x, y, z
        A = block_diag(*[Ax, Ay, Az, Ayaw])
        b = np.concatenate((bx, by, bz, byaw))

        # Use the neural network to modify the reference
        nn_coeff, pred, nan_encountered = sgd_jax.modify_reference(regularizer, H, A, b, min_snap_coeffs)

        if not nan_encountered:
            c_opt_x = nn_coeff[0:((poly_degree + 1) * m)]
            c_opt_y = nn_coeff[((poly_degree + 1) * m):(2 * (poly_degree + 1) * m)]
            c_opt_z = nn_coeff[(2 * (poly_degree + 1) * m):(3 * (poly_degree + 1) * m)]
            c_opt_yaw = nn_coeff[(3 * (poly_degree + 1) * m):]

    # Return computed coefficients and other necessary data
    return {
        'delta_t': delta_t,
        't_keyframes': t_keyframes,
        'c_opt_x': c_opt_x,
        'c_opt_y': c_opt_y,
        'c_opt_z': c_opt_z,
        'c_opt_yaw': c_opt_yaw,
        'nan_encountered': nan_encountered if use_neural_network else None
    }

# similar to Minsnap class's update function
def update_min_snap(t, min_snap_data):
    # Extract necessary data from min_snap_data
    t_keyframes = min_snap_data['t_keyframes']
    delta_t = min_snap_data['delta_t']
    c_opt_x = min_snap_data['c_opt_x']
    c_opt_y = min_snap_data['c_opt_y']
    c_opt_z = min_snap_data['c_opt_z']
    c_opt_yaw = min_snap_data['c_opt_yaw']
    poly_degree = len(c_opt_x) // len(delta_t) - 1  # Assumes uniform polynomial degree
    yaw_poly_degree = len(c_opt_yaw) // len(delta_t) - 1

    # Initialize output
    x = np.zeros((3,))
    x_dot = np.zeros((3,))
    x_ddot = np.zeros((3,))
    x_dddot = np.zeros((3,))
    x_ddddot = np.zeros((3,))
    yaw = 0
    yaw_dot = 0
    yaw_ddot = 0

    # Find the correct time segment
    segment_index = np.searchsorted(t_keyframes, t, side='right') - 1
    segment_index = max(0, min(segment_index, len(delta_t) - 1))
    t_segment = t - t_keyframes[segment_index]

    # Evaluate the polynomial for each dimension
    for i in range(3):
        coeffs = {
            0: c_opt_x,
            1: c_opt_y,
            2: c_opt_z
        }[i][segment_index * (poly_degree + 1):(segment_index + 1) * (poly_degree + 1)]

        x[i] = np.polyval(np.flip(coeffs), t_segment)
        x_dot[i] = np.polyval(np.polyder(np.flip(coeffs), 1), t_segment)
        x_ddot[i] = np.polyval(np.polyder(np.flip(coeffs), 2), t_segment)
        x_dddot[i] = np.polyval(np.polyder(np.flip(coeffs), 3), t_segment)
        x_ddddot[i] = np.polyval(np.polyder(np.flip(coeffs), 4), t_segment)

    # Evaluate the polynomial for yaw
    yaw_coeffs = c_opt_yaw[segment_index * (yaw_poly_degree + 1):(segment_index + 1) * (yaw_poly_degree + 1)]
    yaw = np.polyval(np.flip(yaw_coeffs), t_segment)
    yaw_dot = np.polyval(np.polyder(np.flip(yaw_coeffs), 1), t_segment)
    yaw_ddot = np.polyval(np.polyder(np.flip(yaw_coeffs), 2), t_segment)

    flat_output = {
        "x": x,
        "x_dot": x_dot,
        "x_ddot": x_ddot,
        "x_dddot": x_dddot,
        "x_ddddot": x_ddddot,
        "yaw": yaw,
        "yaw_dot": yaw_dot,
        "yaw_ddot": yaw_ddot,
    }

    return flat_output


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # Example setup
    waypoints = np.array([[0, 0, 0], [1, 1, 1], [2, 0, 2], [3, 1, 3]])
    yaw_angles = np.array([0, 0, 0, 0])
    poly_degree = 7
    yaw_poly_degree = 7
    v_max = 5
    v_avg = 1
    v_start = [0, 0, 0]
    v_end = [0, 0, 0]
    yaw_rate_max = 2 * np.pi
    use_neural_network = False  # or True, depending on your setup
    regularizer = None  # replace with your regularizer if using neural network

    # Compute min snap coefficients and data
    min_snap_data = compute_min_snap_coefficients(waypoints, yaw_angles, poly_degree, yaw_poly_degree, v_max, v_avg, v_start, v_end, yaw_rate_max, use_neural_network, regularizer)

    # Generate trajectory data?????????
    time = np.linspace(0, 1.1 * min_snap_data['t_keyframes'][-1], 1000)
    trajectories = [update_min_snap(t, min_snap_data) for t in time]

    # Plotting
    x = [traj["x"][0] for traj in trajectories]
    y = [traj["x"][1] for traj in trajectories]
    z = [traj["x"][2] for traj in trajectories]

    plt.figure()
    plt.plot(x, y, label="XY Trajectory")
    plt.scatter(waypoints[:, 0], waypoints[:, 1], color='red', label="Waypoints")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Min Snap Trajectory")

    # Save the plot
    plt.savefig("min_snap_trajectory.png")

    # Optionally, show the plot
    plt.show()
