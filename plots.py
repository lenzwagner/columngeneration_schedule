import seaborn as sns
import matplotlib.pyplot as plt

def plot_obj_val(objValHistRMP):
    sns.set(style='darkgrid')
    sns.scatterplot(x=list(range(len(objValHistRMP))), y=objValHistRMP, marker='o')
    sns.lineplot(x=list(range(len(objValHistRMP))), y=objValHistRMP)
    plt.xlabel('CG Iterations')
    plt.xticks(range(0, len(objValHistRMP)))
    plt.ylabel('Objective function value')
    title = 'Optimal objective value: ' + str(round(objValHistRMP[-1], 2))
    plt.title(title)
    plt.show()

def plot_avg_rc(avg_rc_hist):
    sns.set(style='darkgrid')
    sns.scatterplot(x=list(range(1, len(avg_rc_hist) + 1)), y=avg_rc_hist, marker='o')
    sns.lineplot(x=list(range(1, len(avg_rc_hist) + 1)), y=avg_rc_hist)
    plt.xlabel('CG Iterations')
    plt.xticks(range(1, len(avg_rc_hist)+1))
    plt.ylabel('Reduced Cost')
    title = 'Final reduced cost: ' + str(round(avg_rc_hist[-1], 2))
    plt.title(title)
    plt.show()

def plot_together(objValHistRMP, avg_rc_hist):
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

    plt.show()