"""Plots the data
just run this in pycharm or something, make sure to rename the filenames below for input
"""

import plotly.express as px
import pandas as pd

def read_input(filename: str, arr: list[float]) -> None:
    with open(filename) as f:
        for line in f:
            arr.append(float(line))


def main():
    openblas_runtime = []
    aten_runtime = []
    read_input("outputs/blas_out_full.txt", openblas_runtime) # change these
    read_input("outputs/aten_out_full.txt", aten_runtime)

    data = {"aten": aten_runtime,
            "openblas": openblas_runtime,
            }

    df = pd.DataFrame(data=data)

    # fig = px.bar(x=idx, y=[aten_runtime, openblas_runtime, eigen_runtime], barmode='group', log_y=True)
    fig = px.bar(df, barmode='group', labels={"index": "File Number", "value": "Time"} )
    fig.update_xaxes(tickmode='linear')
    fig.show()


if __name__ == "__main__":
    main()
