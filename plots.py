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
    file_name = f'./images/' + file + '.png'

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
    file_name = f'./images/' + file + '.png'

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
    file_name = f'./images/' + file + '.png'

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
                    color_continuous_scale=[ '#E57373' , '#4B8B9F', '#DAA520' ,'#76B041'])

    fig.update(data=[{'hovertemplate': "Day: %{x}<br>"
                                       "Physician: %{y}<br>"}])

    colors = dict(thickness=35,
                    tickvals=[0, 1, 2, 3],
                    ticktext=['Off', 'Morning', 'Noon', 'Evening'],
                    title = "Shift")

    fig.update(layout_coloraxis_showscale=True, layout_coloraxis_colorbar=colors)


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
    file_name = f'./images/' + file + '.png'

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

def violinplots(list_cg, list_compact):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    df = pd.DataFrame(list_cg, columns=['Time'])
    df1 = pd.DataFrame(list_compact, columns=['Time'])

    sns.violinplot(y=df["Time"], ax=axs[0], color=".8")
    sns.violinplot(y=df1["Time"], ax=axs[1], color=".8")

    median_cg = df["Time"].median()
    median_compact = df1["Time"].median()

    axs[0].axhline(median_cg, color='r', linestyle='--', label='Median')
    axs[0].text(median_cg, axs[0].get_ylim()[1], f'{median_cg}', ha='center', va='top', backgroundcolor='white')

    axs[1].axhline(median_compact, color='r', linestyle='--', label='Median')
    axs[1].text(median_compact, axs[1].get_ylim()[1], f'{median_compact}', ha='center', va='top', backgroundcolor='white')

    axs[0].set_title("Column Generation")
    axs[1].set_title("Compact Solver")

    plt.legend()
    plt.show()


def medianplots(list_cg, list_compact):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    df = pd.DataFrame(list_cg, columns=['Time'])
    df1 = pd.DataFrame(list_compact, columns=['Time'])

    sns.boxplot(x=df["Time"], ax=axs[0])
    axs[0].set_title("Column Generation")

    sns.boxplot(x=df1["Time"], ax=axs[1])
    axs[1].set_title("Compact Solver")

    median_cg = df["Time"].median()
    median_compact = df1["Time"].median()

    axs[0].axvline(median_cg, color='r', linestyle='--', label=f'Median: {median_cg}')
    axs[0].text(0.5, median_cg, f'{median_cg}', ha='center', va='bottom')

    axs[1].axvline(median_compact, color='r', linestyle='--', label=f'Median: {median_compact}')
    axs[1].text(0.5, median_compact, f'{median_compact}', ha='center', va='bottom')

    plt.legend()
    plt.show()

def optBoxplot(vals):
    df = pd.DataFrame(sorted(vals), columns=['Gap'])
    mean_val = np.mean(df)
    plt.axvline(x=mean_val, color='red', linestyle='--', label='Mean')
    sns.boxplot(x=df["Gap"])
    plt.title("Optimality Gap in %")
    plt.show()

def pie_chart(optimal):
    zeros = sum(value == 0 for value in optimal.values())
    ones = sum(value == 1 for value in optimal.values())

    data = pd.DataFrame({'Category': ['Yes', 'No'], 'Count': [ones, zeros]})

    plt.figure(figsize=(6, 6))
    plt.pie(data['Count'], labels=data['Category'], colors=['#008fd5', '#fc4e07'], startangle=90, autopct='%1.1f%%')

    plt.ylabel('')
    plt.xlabel('')
    plt.title("Optimality Distribution")
    plt.legend(labels=['Yes', 'No'], loc='lower right', bbox_to_anchor=(1.0, 0.3), title = "Optimal Solution?")

    plt.show()