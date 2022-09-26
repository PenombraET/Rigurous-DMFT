from mpi4py import MPI
import utils as ut
import DMFT_iteration as it

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    ut.set_seed(comm)

    parameters = ut.init_parameters()
    parameters["m_0"] = 0.025
    parameters["alpha"] = 0.9
    parameters["b"] = 1.0
    parameters["lambd"] = 1.0
    parameters["dt"] = 0.04
    parameters["T"] = 50
    parameters["n_samples"] = 10000
    parameters["damping"] = 0.55
    parameters["n_iterations"] = 30

    ut.clean_log(comm)
    ut.printMPI(f"Initialising...", comm)
    if rank == 0:
        it.init(parameters)
    comm.Barrier()

    n_iterations = int(parameters["n_iterations"])
    for iteration in range(n_iterations):
        ut.printMPI(f"Iteration {iteration+1} of {n_iterations}", comm)
        it.iterate(comm, iteration, parameters)
        comm.Barrier()
