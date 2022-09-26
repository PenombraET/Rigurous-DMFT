import utils_plot as utp
import matplotlib.pyplot as plt
import pandas as pd
import utils as ut


parameters = ut.init_parameters()
parameters["m_0"] = 0.025
parameters["D"] = 2000
parameters["alpha"] = 0.9
parameters["b"] = 1.0
parameters["lambd"] = 1.0
parameters["dt"] = 0.04
parameters["T"] = 50

iter_start = 20
iter_end = 30

utp.plot_magnetisation_simulations(parameters)
for iteration in range(iter_start, iter_end):
    utp.plot_magnetisation(parameters, iteration)


plt.legend()
plt.xlabel("Time")
plt.ylabel("Magnetisation")
plt.xscale("log")
plt.savefig("comparison.png")
plt.show()
