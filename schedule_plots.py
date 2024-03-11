import pandas as pd
import plotly.express as px

# DF
dic = {(1, 1, 1): 1.0, (1, 1, 2): 0.0, (1, 1, 3): 0.0, (1, 2, 1): 0.0, (1, 2, 2): 1.0, (1, 2, 3): 0.0, (1, 3, 1): 0.0, (1, 3, 2): 0.0, (1, 3, 3): 0.0, (1, 4, 1): 0.0, (1, 4, 2): 1.0, (1, 4, 3): 0.0, (1, 5, 1): 1.0, (1, 5, 2): 0.0, (1, 5, 3): 0.0, (1, 6, 1): 0.0, (1, 6, 2): 1.0, (1, 6, 3): 0.0, (1, 7, 1): 0.0, (1, 7, 2): 1.0, (1, 7, 3): 0.0, (2, 1, 1): 1.0, (2, 1, 2): 0.0, (2, 1, 3): 0.0, (2, 2, 1): 1.0, (2, 2, 2): 0.0, (2, 2, 3): 0.0, (2, 3, 1): 1.0, (2, 3, 2): 0.0, (2, 3, 3): 0.0, (2, 4, 1): 0.0, (2, 4, 2): 0.0, (2, 4, 3): 0.0, (2, 5, 1): 1.0, (2, 5, 2): 0.0, (2, 5, 3): 0.0, (2, 6, 1): 0.0, (2, 6, 2): 0.0, (2, 6, 3): 1.0, (2, 7, 1): 0.0, (2, 7, 2): 1.0, (2, 7, 3): 0.0, (3, 1, 1): 1.0, (3, 1, 2): 0.0, (3, 1, 3): 0.0, (3, 2, 1): 0.0, (3, 2, 2): 1.0, (3, 2, 3): 0.0, (3, 3, 1): 0.0, (3, 3, 2): 0.0, (3, 3, 3): 0.0, (3, 4, 1): 1.0, (3, 4, 2): 0.0, (3, 4, 3): 0.0, (3, 5, 1): 1.0, (3, 5, 2): 0.0, (3, 5, 3): 0.0, (3, 6, 1): 1.0, (3, 6, 2): 0.0, (3, 6, 3): 0.0, (3, 7, 1): 0.0, (3, 7, 2): 1.0, (3, 7, 3): 0.0}

out = (pd.Series(dic, name='worked')
         .rename_axis(['Physician', 'Day', 'Shift'])
         .reset_index()
         .assign(shift=lambda x: x['Shift'].where(x['worked'].eq(1), 0))
         .pivot_table(index='Physician', columns='Day',
                      values='Shift', aggfunc='sum')
       )

df = pd.DataFrame(out)
print(df)

def schedulePlot(df, undercoverage):
    fig = px.scatter(df, x='Day', y='Physician', color='Shift', symbol='Shift',
                     color_discrete_sequence=px.colors.qualitative.Set3,
                     symbol_sequence=['square', 'diamond', 'circle'],
                     title=f'Undercoverage: {undercoverage}')

    fig.update_traces(marker=dict(size=30), selector=dict(type='scatter'))
    colorbar = dict(thickness=25,
                    tickvals=[0, 1, 2, 3],
                    ticktext=['Day', 'Evening', 'Night', 'Off'])
    fig.update(layout_coloraxis_showscale=True, layout_coloraxis_colorbar=colorbar)
    fig.show()

    return fig


schedulePlot(df, 0.2)
