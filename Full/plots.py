import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def violinplots(list_cg, list_compact):
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
    plt.pie(data['Count'], labels=data['Category'], colors=['#F18F01', '#048BA8'], startangle=90, autopct='%1.1f%%')

    plt.ylabel('')
    plt.xlabel('')
    plt.title("Optimality Distribution")
    plt.legend(labels=['Yes', 'No'], loc='lower right', bbox_to_anchor=(1.0, 0.3), title = "Optimal Solution?")

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

    axs[0].axvline(median_cg, color='r', linestyle='--', label='Median')
    axs[0].text(median_cg, axs[0].get_ylim()[1], f'{median_cg}', ha='center', va='top', backgroundcolor='white')

    axs[1].axvline(median_compact, color='r', linestyle='--', label='Median')
    axs[1].text(median_compact, axs[1].get_ylim()[1], f'{median_compact}', ha='center', va='top', backgroundcolor='white')

    plt.legend()
    plt.show()

def performancePlot(p_list, days):
    sns.set(style='darkgrid')

    phys_nr = len(p_list) // days

    phys_list = [p_list[i * days:(i + 1) * days] for i in range(phys_nr)]

    data = {'Day': list(range(1, days + 1))}
    for idx, phys in enumerate(phys_list):
        data[f'Phys {idx + 1}'] = phys

    df = pd.DataFrame(data)
    df_melted = df.melt('Day', var_name='Phys', value_name='Performance')

    palette = sns.color_palette("rocket", phys_nr)
    plt.xticks(range(1, days + 1))

    sns.lineplot(data=df_melted, x='Day', y='Performance', hue='Phys', style='Phys', markers=True, dashes=False, alpha=0.8, palette=palette)
    plt.xlabel('Day')
    plt.ylabel('Motivation')
    plt.title('Performance over Time')
    plt.legend(title='Phys')

    plt.show()


def plot_obj_val(objValHistRMP, name):
    file = str(name)
    file_name = f'./images/' + file + '.png'

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

    plt.savefig(file_name, format='png')
    plt.show()

def plot_avg_rc(avg_rc_hist, name):
    file = str(name)
    file_name = f'./images/' + file + '.png'

    sns.set(style='darkgrid')
    sns.scatterplot(x=list(range(1, len(avg_rc_hist) + 1)), y=avg_rc_hist, marker='o', color='#3c4cad')
    sns.lineplot(x=list(range(1, len(avg_rc_hist) + 1)), y=avg_rc_hist, color='#3c4cad')
    plt.xlabel('Iterations')
    plt.xticks(range(1, len(avg_rc_hist)+1))
    plt.ylabel('Reduced Cost')
    title = 'Final reduced cost: ' + str(round(avg_rc_hist[-1], 2))
    plt.title(title)

    plt.savefig(file_name, format='png')
    plt.show()