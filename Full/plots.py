import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy

def violinplots(list_cg, list_compact, name):
    file_dir = f'.\images'
    file_name = str(name) + '.png'
    plot_path = os.path.join(file_dir, file_name)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    df = pd.DataFrame(list_cg, columns=['Time'])
    df1 = pd.DataFrame(list_compact, columns=['Time'])

    sns.violinplot(x=df["Time"], ax=axs[0], color=".8", bw_adjust=.5, inner_kws=dict(box_width=15, whis_width=2, color=".8"))
    sns.violinplot(x=df1["Time"], ax=axs[1], color=".8", bw_adjust=.5, inner_kws=dict(box_width=15, whis_width=2, color=".8"))

    median_cg = df["Time"].median()
    median_compact = df1["Time"].median()

    axs[0].axvline(median_cg, color='r', linestyle='--', label='Median')
    axs[0].text(median_cg, axs[0].get_ylim()[1], f'{median_cg}', ha='center', va='top', backgroundcolor='white')

    axs[1].axvline(median_compact, color='r', linestyle='--', label='Median')
    axs[1].text(median_compact, axs[1].get_ylim()[1], f'{median_compact}', ha='center', va='top', backgroundcolor='white')

    axs[0].set_title("Column Generation")
    axs[1].set_title("Compact Solver")

    plt.legend()
    plt.savefig(plot_path, format='png')

    plt.show()

def optBoxplot(vals, name):
    file_dir = f'.\images'
    file_name = str(name) + '.png'
    plot_path = os.path.join(file_dir, file_name)

    df = pd.DataFrame(sorted(vals), columns=['Gap'])
    mean_val = np.mean(df)
    plt.axvline(x=mean_val, color='red', linestyle='--', label='Mean')
    sns.boxplot(x=df["Gap"])
    plt.title("Optimality Gap in %")
    plt.savefig(plot_path, format='png')

    plt.show()

def pie_chart(optimal, name):
    file_dir = f'.\images'
    file_name = str(name) + '.png'
    plot_path = os.path.join(file_dir, file_name)

    zeros = sum(value == 0 for value in optimal.values())
    ones = sum(value == 1 for value in optimal.values())

    data = pd.DataFrame({'Category': ['Yes', 'No'], 'Count': [ones, zeros]})

    plt.figure(figsize=(6, 6))
    plt.pie(data['Count'], labels=data['Category'], colors=['#F18F01', '#048BA8'], startangle=90, autopct='%1.1f%%')

    plt.ylabel('')
    plt.xlabel('')
    plt.title("Optimality Distribution")
    plt.legend(labels=['Yes', 'No'], loc='lower right', bbox_to_anchor=(1.0, 0.3), title = "Optimal Solution?")
    plt.savefig(plot_path, format='png')

    plt.show()

def medianplots(list_cg, list_compact, name):
    file_dir = f'.\images'
    file_name = str(name) + '.png'
    plot_path = os.path.join(file_dir, file_name)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    df = pd.DataFrame(list_cg, columns=['Time'])
    df1 = pd.DataFrame(list_compact, columns=['Time'])

    sns.boxplot(x=df["Time"], ax=axs[0])
    axs[0].set_title("Column Generation")

    sns.boxplot(x=df1["Time"], ax=axs[1])
    axs[1].set_title("Compact Solver")

    median_cg = df["Time"].median()
    median_compact = df1["Time"].median()

    axs[0].axvline(median_cg, color='r', linestyle='--', label='Median')
    axs[0].text(median_cg, axs[0].get_ylim()[1], f'{median_cg}', ha='center', va='top', backgroundcolor='white')

    axs[1].axvline(median_compact, color='r', linestyle='--', label='Median')
    axs[1].text(median_compact, axs[1].get_ylim()[1], f'{median_compact}', ha='center', va='top', backgroundcolor='white')
    plt.legend()

    plt.savefig(plot_path, format='png')
    plt.show()

def performancePlot(ls, days, phys_nr, name):
    sns.set(style='darkgrid')

    file_dir = f'.\images'
    file_name = str(name) + '.png'
    plot_path = os.path.join(file_dir, file_name)


    grid = list(range(1, days + 1))
    graphs = [ls[i:i+days] for i in range(0, len(ls), 14)]

    fig, ax = plt.subplots()

    lw = 1.5
    palette = sns.color_palette("rocket", phys_nr)
    for gg, graph in enumerate(graphs, start=1):
        trans_offset = offset_copy(ax.transData, fig=fig, x=lw * gg, y=lw * gg, units='dots')
        ax.plot(grid, graph, lw=lw, transform=trans_offset, label=gg, color=palette[gg-1], alpha = 0.6)

    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.35), title='Physician')
    # manually set the axes limits, because the transform doesn't set them automatically
    ax.set_xlim(grid[0] - .5, grid[-1] + .5)
    ax.set_ylim(min([min(g) for g in graphs]) - .02, max([max(g) for g in graphs]) + .02)

    plt.xlabel('Day')
    plt.ylabel('Performance')
    plt.title('Physician Performance over Time')
    plt.xticks(range(1, days + 1))

    plt.savefig(plot_path, format='png')

    plt.show()


def plot_obj_val(objValHistRMP, name):
    file_dir = f'.\images'
    file_name = str(name) + '.png'
    plot_path = os.path.join(file_dir, file_name)

    sns.set(style='darkgrid')
    sns.scatterplot(x=list(range(len(objValHistRMP[:-1]))), y=objValHistRMP[:-1], marker='o', color='#3c4cad',
                    label='Objective Value')
    sns.lineplot(x=list(range(len(objValHistRMP))), y=objValHistRMP, color='#3c4cad')
    sns.scatterplot(x=[len(objValHistRMP) - 1], y=[objValHistRMP[-1]], color='#f9c449', s=100, label='Last Point')

    plt.xlabel('Iterations')
    plt.xticks(range(0, len(objValHistRMP)))
    plt.ylabel('Objective function value')
    title = 'Optimal integer objective value: ' + str(round(objValHistRMP[-1], 2))
    plt.title(title)

    x_ticks_labels = list(range(len(objValHistRMP) - 1)) + ["Int. Solve"]
    plt.xticks(ticks=list(range(len(objValHistRMP))), labels=x_ticks_labels)

    h, l = plt.gca().get_legend_handles_labels()
    plt.legend(h[:2], l[:2] + ['Last Point'], loc='best', handletextpad=0.1, handlelength=1, fontsize='medium',
               title='Legend')

    plt.savefig(plot_path, format='png')
    plt.show()

def plot_avg_rc(avg_rc_hist, name):
    file_dir = f'.\images'
    file_name = str(name) + '.png'
    plot_path = os.path.join(file_dir, file_name)

    sns.set(style='darkgrid')
    sns.scatterplot(x=list(range(1, len(avg_rc_hist) + 1)), y=avg_rc_hist, marker='o', color='#3c4cad')
    sns.lineplot(x=list(range(1, len(avg_rc_hist) + 1)), y=avg_rc_hist, color='#3c4cad')
    plt.xlabel('Iterations')
    plt.xticks(range(1, len(avg_rc_hist)+1))
    plt.ylabel('Reduced Cost')
    title = 'Final reduced cost: ' + str(round(avg_rc_hist[-1], 2))
    plt.title(title)

    plt.savefig(plot_path, format='png')
    plt.show()