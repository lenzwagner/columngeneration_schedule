import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import os
import matplotlib.pyplot as plt

def createDir():
    if not os.path.exists("images"):
        os.mkdir("images")

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
                    color_continuous_scale=["purple", "orange", "yellow", 'pink'])

    fig.update(data=[{'hovertemplate': "Day: %{x}<br>"
                                       "Physician: %{y}<br>"}])

    colorbar = dict(thickness=35,
                    tickvals=[0, 1, 2, 3],
                    ticktext=['Off', 'Evening', 'Noon', 'Morning'])

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

    fig.update_layout(
        xaxis=dict(
            showgrid=True,
            gridwidth=1.5,
            gridcolor='LightGray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1.5,
            gridcolor='LightGray'
        )
    )

    fig.show()
    return fig