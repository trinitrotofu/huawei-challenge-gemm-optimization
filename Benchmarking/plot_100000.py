"""Plots the data
read plot_blas_aten
It's the same thing but also plots eigen
"""

import plotly.express as px
import pandas as pd

def read_input(filename: str, arr: list[float]) -> None:
    with open(filename) as f:
        for line in f:
            arr.append(float(line))


def main():
    openblas_runtime = []
    eigen_runtime = []
    aten_runtime = []
    read_input("outputs/blas_out.txt", openblas_runtime)
    read_input("outputs/eigen_out.txt", eigen_runtime)
    read_input("outputs/aten_out.txt", aten_runtime)

    data = {"aten": aten_runtime,
            "openblas": openblas_runtime,
            "eigen": eigen_runtime
            }

    df = pd.DataFrame(data=data)

    # fig = px.bar(x=idx, y=[aten_runtime, openblas_runtime, eigen_runtime], barmode='group', log_y=True)
    fig = px.bar(df, barmode='group', log_y=True, labels={"index": "File Number (first 100,000 calls)", "value": "Time (log scale)"} )
    fig.update_xaxes(tickmode='linear')
    fig.show()



if __name__ == "__main__":
    main()
