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

a = sns.jointplot(
    x="MTO_PIA",
    y="DEPARTAMENTO",
    data=data_df,
    size=8,
    alpha=0.6,
    color="k",
    marker="x",
)
b = sns.jointplot(
    x="MTO_PIA",
    y="PROVINCIA",
    data=data_df,
    size=8,
    alpha=0.6,
    color="k",
    marker="x",
)
c = sns.jointplot(
    x="MTO_PIA",
    y="UBIGEO",
    data=data_df,
    size=8,
    alpha=0.6,
    color="k",
    marker="x",
)


plt.show()
