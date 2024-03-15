import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def violinplots(list_cg, list_compact):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    df = pd.DataFrame(list_cg, columns=['Time'])
    df1 = pd.DataFrame(list_compact, columns=['Time'])

    sns.violinplot(x=df["Time"], ax=axs[0], inner_kws=dict(box_width=15, whis_width=2, color=".8"))
    axs[0].set_title("Column Generation")

    sns.violinplot(x=df1["Time"], ax=axs[1], inner_kws=dict(box_width=15, whis_width=2, color=".8"))
    axs[1].set_title("Compact Solver")

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