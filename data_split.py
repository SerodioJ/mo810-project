import argparse
import os
from pathlib import Path

import numpy as np
import dask.dataframe as ddf
from sklearn.model_selection import train_test_split


def data_split(args):
    df = ddf.read_parquet(os.path.join(args.path, "dataframe.parquet"))
    indices = np.arange(len(df))
    remaining, validation = train_test_split(indices, test_size=0.1)
    train, test = train_test_split(remaining, test_size=0.25)
    np.save(os.path.join(args.path, f"{args.prefix}_train_indices"), train)
    np.save(os.path.join(args.path, f"{args.prefix}_test_indices"), test)
    np.save(os.path.join(args.path, f"{args.prefix}_validation_indices"), validation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--path", help="path to data", type=Path, default="data")

    parser.add_argument(
        "-f", "--prefix", help="prefix for generated split", type=str, default="split"
    )

    args = parser.parse_args()

    data_split(args)
