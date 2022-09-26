import numpy as np
import utils as ut
from numba import njit
from mpi4py import MPI
import loss_functions as lf


@njit(fastmath=True)
def make_batch_sim(parameters):
    """Makes batch selection variable for SGD

    Args:
        parameters (dict): parameter dictionary

    Returns:
        np.ndarray: [T, N] matrix, sample selection variable
    """

    T = int(parameters["T"])
    n_samples = int(parameters["N"])
    b = parameters["b"]

    batch = (np.random.rand(T, n_samples)) - b
    for t in range(T):
        for sample in range(n_samples):
            if batch[t, sample] < 0:
                batch[t, sample] = 1 / b
            else:
                batch[t, sample] = 0
    return batch


@njit(fastmath=True)
def simulation(parameters):
    """Simulate the gradient descent dynamics directly

    Args:
        parameters (dict): parameter dictionary

    Returns:
        np.ndarray: [n_sim, T] matrix, n_sim simulation of the magnetisation at times T
    """

    n_sim = int(parameters["n_sim"])
    T = int(parameters["T"])
    D = int(parameters["D"])
    N = int(parameters["N"])
    dt = parameters["dt"]
    lambd = parameters["lambd"]
    m_0 = parameters["m_0"]

    m = np.zeros((n_sim, T))
    norm = np.zeros((n_sim, T))

    for iter in range(n_sim):
        X = np.random.normal(0, 1, (D, N))
        batch = make_batch_sim(parameters)
        weight_opt = np.random.normal(0, 1, D)
        h_opt = np.sign(weight_opt @ X / np.sqrt(D))

        weight = m_0 * weight_opt + np.random.normal(0, 1, D)

        m[iter, 0] = weight @ weight_opt / D

        grad_loss = np.zeros(D)
        for t in range(T):
            h = np.sign(weight @ X / np.sqrt(D))
            for d in range(D):
                grad_loss[d] = np.sum(lf.d_nu(h, h_opt) * X[d] * batch[t] / np.sqrt(D))
            weight = weight - dt * lambd * weight - dt * grad_loss

            m[iter, t] = weight @ weight_opt / D
            norm[iter, t] = weight @ weight / D
    return m


if __name__ == "__main__":
    parameters = ut.init_parameters()
    parameters["m_0"] = 0.025
    parameters["D"] = 2000
    parameters["alpha"] = 0.9
    parameters["N"] = parameters["alpha"] * parameters["D"]
    parameters["b"] = 1.0
    parameters["lambd"] = 1.0
    parameters["dt"] = 0.04
    parameters["T"] = 50
    parameters["n_sim"] = 3

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ut.set_seed(comm)
    m = simulation(parameters)

    if rank == 0:
        m_comm = np.zeros((size, int(parameters["n_sim"]), int(parameters["T"])))
    else:
        # We don't need them so let's save memory
        m_comm = 0

    comm.Gather(m, m_comm, root=0)

    if rank == 0:
        m_comm = m_comm.reshape((1, -1, int(parameters["T"])))[0]
        ut.save_sim(m_comm.T, parameters)
