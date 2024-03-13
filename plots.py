import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
from matplotlib.ticker import PercentFormatter, MaxNLocator
import itertools
import os
import matplotlib.pyplot as plt

def createDir():
    if not os.path.exists("images"):
        os.mkdir("images")

def plot_obj_val(objValHistRMP, name):
    file = str(name)
    file_name = f'G:/Meine Ablage/Doktor/Dissertation/Paper 1/Data/Pics/' + file + '.png'

    sns.set(style='darkgrid')
    sns.scatterplot(x=list(range(len(objValHistRMP))), y=objValHistRMP, marker='o')
    sns.lineplot(x=list(range(len(objValHistRMP))), y=objValHistRMP)
    plt.xlabel('CG Iterations')
    plt.xticks(range(0, len(objValHistRMP)))
    plt.ylabel('Objective function value')
    title = 'Optimal objective value: ' + str(round(objValHistRMP[-1], 2))
    plt.title(title)
    plt.savefig(file_name, format='png')
    plt.show()

def plot_avg_rc(avg_rc_hist, name):
    file = str(name)
    file_name = f'G:/Meine Ablage/Doktor/Dissertation/Paper 1/Data/Pics/' + file + '.png'

    sns.set(style='darkgrid')
    sns.scatterplot(x=list(range(1, len(avg_rc_hist) + 1)), y=avg_rc_hist, marker='o')
    sns.lineplot(x=list(range(1, len(avg_rc_hist) + 1)), y=avg_rc_hist)
    plt.xlabel('CG Iterations')
    plt.xticks(range(1, len(avg_rc_hist)+1))
    plt.ylabel('Reduced Cost')
    title = 'Final reduced cost: ' + str(round(avg_rc_hist[-1], 2))
    plt.title(title)
    plt.savefig(file_name, format='png')
    plt.show()

def plot_together(objValHistRMP, avg_rc_hist, name):
    file = str(name)
    file_name = f'G:/Meine Ablage/Doktor/Dissertation/Paper 1/Data/Pics/' + file + '.png'


    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.scatterplot(x=list(range(len(objValHistRMP))), y=objValHistRMP, marker='o', ax=axs[0])
    sns.lineplot(x=list(range(len(objValHistRMP))), y=objValHistRMP, ax=axs[0])
    axs[0].set_xlabel('CG Iterations')
    axs[0].set_xticks(range(0, len(objValHistRMP)))
    axs[0].set_ylabel('Objective function value')
    title = 'Optimal objective value: ' + str(round(objValHistRMP[-1], 2))
    axs[0].set_title(title)

    sns.scatterplot(x=list(range(1, len(avg_rc_hist) + 1)), y=avg_rc_hist, marker='o', ax=axs[1])
    sns.lineplot(x=list(range(1, len(avg_rc_hist) + 1)), y=avg_rc_hist, ax=axs[1])
    axs[1].set_xlabel('CG Iterations')
    axs[1].set_xticks(range(1, len(avg_rc_hist)+1))
    axs[1].set_ylabel('Reduced Cost')
    title = 'Final reduced cost: ' + str(round(avg_rc_hist[-1], 2))
    axs[1].set_title(title)

    plt.savefig(file_name, format='png')
    plt.show()

def visualize_schedule(dic, days, undercoverage):
    s = pd.Series(dic)

    data = (s.loc[lambda s: s == 1]
           .reset_index(-1)['level_2'].unstack(fill_value=0)
           .reindex(index=s.index.get_level_values(0).unique(),
                    columns=s.index.get_level_values(1).unique(),
                    fill_value=0
                    )
           )

    data.index = data.index.astype(int)
    data.columns = data.columns.astype(str)

    title_str = f'Physician Schedules | Total Undercoverage: {undercoverage}'
    fig = px.imshow(data[[str(i) for i in range(1, days + 1)]],
                    color_continuous_scale=['#264653', '#2A9D8F', '#E9C46A', '#E76F51'])

    fig.update(data=[{'hovertemplate': "Day: %{x}<br>"
                                       "Physician: %{y}<br>"}])

    colorbar = dict(thickness=35,
                    tickvals=[0, 1, 2, 3],
                    ticktext=['Off', 'Evening', 'Noon', 'Morning'],
                    title = "Shift-Types")

    fig.update(layout_coloraxis_showscale=True, layout_coloraxis_colorbar=colorbar)


    x_ticks = np.arange(1, days + 1)
    day_labels = ['Day ' + str(i) for i in x_ticks]
    fig.update_xaxes(tickvals=x_ticks, ticktext=day_labels)

    y_ticks = np.arange(1, data.shape[0] + 1)
    physician_labels = ['Physician ' + str(i) for i in y_ticks]
    fig.update_yaxes(tickvals=y_ticks, ticktext=physician_labels)

    fig.update_layout(
        title={
            'text': title_str,
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        }
    )

    fig.show()
    return fig

def combine_legends(*axes):
    handles = list(itertools.chain(*[ax.get_legend_handles_labels()[0] for ax in axes]))
    labels = list(
        itertools.chain(*[ax3.get_legend_handles_labels()[1] for ax3 in axes])
    )
    return handles, labels


def set_obj_axes_labels(ax):
    ax.set_ylabel("Objective value")
    ax.set_xlabel("Iterations")


def plot_obj(df, ax):
    ax.step(
        list(range(len(df))),
        df,
        where="post",
        color="b",
        label="Obj",
    )
    set_obj_axes_labels(ax)

def plot_gap(df1, ax):
    ax.step(
        list(range(len(df1))),
        df1,
        where="post",
        color="green",
        label="Gap",
    )
    ax.set_ylabel("Optimality Gap in %")
    ax.set_ylim(0, 1)
    formatter = PercentFormatter(1)
    ax.yaxis.set_major_formatter(formatter)


def optimalityplot(df, df2, last_itr, name):
    file = str(name)
    file_name = f'G:/Meine Ablage/Doktor/Dissertation/Paper 1/Data/Pics/' + file + '.png'

    with plt.style.context("seaborn-v0_8"):
        _, ax = plt.subplots(figsize=(8, 5))

        plot_obj(df, ax)

        ax2 = ax.twinx()
        plot_gap(df2, ax2)

        ax.set_xlim(0, last_itr-2)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        print(combine_legends(ax, ax2))
        ax.legend(*combine_legends(ax, ax2))

        plt.savefig(file_name, format='png')

        plt.show()
