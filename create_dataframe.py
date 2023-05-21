import argparse
import os
import csv
from pathlib import Path

import numpy as np
import dask.dataframe as ddf


def get_window_columns(w_hw):
    order = ["x", "y", "z"]
    columns = []
    for source, hws in w_hw.items():
        for i, hw in enumerate(hws):
            for ind in reversed(range(hw)):
                columns.append(f"-{ind+1}{order[i]}_{source}")
            for ind in range(hw):
                columns.append(f"{ind+1}{order[i]}_{source}")
    return columns


def generate_csv_rows(data, w_hw, shape, columns):
    with open("dataframe.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for x in range(shape[0]):
            print(x)
            for y in range(shape[1]):
                for z in range(shape[2]):
                    row = [
                        (x * shape[1] + y) * shape[2] + z,
                        x,
                        y,
                        z,
                        data["raw"][x, y, z],
                        data["env"][x, y, z],
                        data["cos"][x, y, z],
                        data["freq"][x, y, z],
                        data["cluster"][x, y, z],
                    ]
                    for source, hws in w_hw.items():
                        views = [
                            data[source][:, y, z],
                            data[source][x, :, z],
                            data[source][x, y, :],
                        ]
                        for base, view, hw in zip([x, y, z], views, hws):
                            for n_offset in reversed(range(1, hw + 1)):
                                ind = base - n_offset
                                if ind < 0:
                                    row.append(np.nan)
                                else:
                                    row.append(view[ind])
                            for p_offset in range(1, hw + 1):
                                ind = base + p_offset
                                if ind < view.shape[0]:
                                    row.append(view[ind])
                                else:
                                    row.append(np.nan)
                    writer.writerow(row)


def data_load(args):
    files = {
        "raw": "F3_train.npy",
        "env": "F3_train_envelope.npy",
        "cos": "F3_train_cosine_instantaneous_phase.npy",
        "freq": "F3_train_instantaneous_frequency.npy",
        "cluster": "F3_train_labels.npy",
    }

    window_hw = {
        "raw": (1, 1, 3),
        "env": (1, 1, 1),
        "cos": (1, 1, 1),
        "freq": (1, 1, 1),
        "cluster": (0, 0, 0),
    }
    print("Loading NumPy data...")
    data = {
        source: np.load(os.path.join(args.path, file)) for source, file in files.items()
    }
    columns = [
        "index",
        "x",
        "y",
        "z",
        "raw",
        "env",
        "cos",
        "freq",
        "cluster",
    ] + get_window_columns(window_hw)
    print("Creating CSV file...")
    generate_csv_rows(data, window_hw, data["raw"].shape, columns)

    print("Loading Dasf DataFrame...")
    df = ddf.read_csv("dataframe.csv")
    print("Saving DataFrame in Parquet format...")
    df.to_parquet("dataframe.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--path", help="path to data", type=Path, default="data")

    args = parser.parse_args()

    data_load(args)
