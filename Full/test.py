import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import itertools
import os
import gurobi_logtools as glt


def combine_legends(*axes):
    handles = list(itertools.chain(*[ax.get_legend_handles_labels()[0] for ax in axes]))
    labels = list(
        itertools.chain(*[ax.get_legend_handles_labels()[1] for ax in axes])
    )
    return handles, labels


def set_obj_axes_labels(ax):
    ax.set_ylabel("Objective value")
    ax.set_xlabel("Time")


def plot_incumbent(df, ax):
    ax.step(
        df["Time"],
        df["Incumbent"],
        where="post",
        color="b",
        label="Incumbent",
    )
    set_obj_axes_labels(ax)


def plot_bestbd(df, ax):
    ax.step(
        df["Time"],
        df["BestBd"],
        where="post",
        color="r",
        label="BestBd",
    )
    set_obj_axes_labels(ax)


def plot_fillabsgap(df, ax):
    ax.fill_between(
        df["Time"],
        df["BestBd"],
        df["Incumbent"],
        step="post",
        color="grey",
        alpha=0.3,
    )
    set_obj_axes_labels(ax)


def plot_relgap(df, ax):
    ax.step(
        df["Time"],
        df["Gap"],
        where="post",
        color="green",
        label="Gap",
    )
    ax.set_ylabel("Optimailty Gap in %")
    ax.set_ylim(0, 1)
    formatter = PercentFormatter(1)
    ax.yaxis.set_major_formatter(formatter)


def plot(df, limit, name):
    file = str(name)
    file_name = f'.{os.sep}images{os.sep}{file}.png'
    with plt.style.context("seaborn-v0_8"):
        _, ax = plt.subplots(figsize=(8, 5))

        plot_incumbent(df, ax)
        plot_bestbd(df, ax)
        plot_fillabsgap(df, ax)

        ax2 = ax.twinx()
        plot_relgap(df, ax2)

        ax.set_xlim(1,limit)
        ax.legend(*combine_legends(ax, ax2))
        plt.savefig(file_name, format='png')

        plt.show()