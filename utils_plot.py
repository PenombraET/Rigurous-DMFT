import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils as ut


def get_standard_color(i):
    """Obtain the color of the standars matplotlib color cycle.

    Args:
        i (int): order in the cycle. Accepts any integer

    Returns:
        str: color identifier
    """

    return plt.rcParams["axes.prop_cycle"].by_key()["color"][i % 10]


def plot_magnetisation(parameters, iteration):
    """Plot the magnetisation curve at the "iteration" iteration

    Args:
        parameters (dict): parameters dictionary
        iteration (int): iteration index, 0 is the starting ansatz
    """

    data = ut.load_data(parameters, iteration)
    if iteration == 0:
        label = "Starting state"
        color = "black"
    else:
        label = f"Iter = {iteration}"
        color = get_standard_color(iteration - 1)

    plt.plot(np.linspace(data["dt"], data["T"] * data["dt"], data["T"]), data["m"], lw=1.5, alpha=0.5, marker="", label=label, color=color)


def plot_magnetisation_simulations(parameters):
    """Plot the magnetisation curve from a simulation

    Args:
        parameters (dict): parameters dictionary
    """

    dt = parameters["dt"]
    T = parameters["T"]

    m = ut.load_sim(parameters)
    plt.errorbar(np.linspace(dt, T * dt, int(T)), m.mean(axis=1), m.std(axis=1), marker=".", label="Simulation", color="red")
