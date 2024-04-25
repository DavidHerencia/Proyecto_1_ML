#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "./datasets"]).decode("utf8"))

data_df = pd.read_csv("./datasets/train.csv")

print(data_df.head())


def scatter(column, r_limit=None):
    fig, scatter = plt.subplots(figsize=(10, 6), dpi=100)
    scatter = sns.scatterplot(x="MTO_PIA", y=column, data=data_df)

    scatter.set_xlim(right=r_limit)

    plt.show()


def scatter_log(column):
    fig, scatter = plt.subplots(figsize=(10, 6), dpi=100)
    scatter = sns.scatterplot(x="MTO_PIA", y=column, data=data_df)
    plt.xscale("log")

    plt.show()


def joint_plot(column):
    a = sns.jointplot(
        x="MTO_PIA",
        y=column,
        data=data_df,
        size=8,
        alpha=0.6,
        color="k",
        marker="x",
    )
    plt.show()


column = "ESPECIFICA_DET"
# joint_plot(column)
# scatter(column, 1000000)
scatter_log(column)
# scatter(column)
