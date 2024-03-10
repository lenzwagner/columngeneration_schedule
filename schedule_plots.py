import pandas as pd
import random
import plotly.express as px

# DF
data = {'Physician': [], 'Day': [], 'Shift': []}
nr_phys = 10
nr_days = 7
nr_shifts = 6

for i in range(1, nr_phys + 1):
    for j in range(1, nr_days + 1):
        data['Physician'].append(i)
        data['Day'].append(j)
        data['Shift'].append(None)

max_days= 5
shifts_day = nr_phys // max_days

for i in range(nr_days):
    physicians = list(range(nr_phys))
    random.shuffle(physicians)
    for j in range(shifts_day):
        for k in range(max_days):
            physician = physicians[j * max_days + k]
            data['Shift'][i * nr_phys + physician] = j

df = pd.DataFrame(data)


def schedulePlot(df, undercoverage):
    fig = px.scatter(df, x='Day', y='Physician', color='Shift', symbol='Shift',
                     color_discrete_sequence=px.colors.qualitative.Set3,
                     symbol_sequence=['square', 'diamond', 'circle'],
                     title=f'Undercoverage: {undercoverage}')

    fig.update_traces(marker=dict(size=30), selector=dict(type='scatter'))

    fig.show()

    return fig


schedulePlot(df, 0.2)
