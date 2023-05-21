import argparse
import itertools
import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import dask.dataframe as ddf
import pandas as pd


class TrainingSetup:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("command", choices=["exploration_space", "data_partition"])
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Invalid command")
            parser.print_help()
            exit(1)

        getattr(self, args.command)()

    def exploration_space(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-p",
            "--path",
            help="path to model folder",
            type=Path,
            nargs="+",
            required=True,
        )
        parser.add_argument(
            "-m",
            "--model_type",
            help="type of model, pre-definied is for pre-defined sklearn models and, ensemble for custom ensembles",
            type=str,
            choices=["pre-defined", "ensemble"],
            default="pre-defined",
        )
        args = parser.parse_args(sys.argv[2:])

        for path in args.path:
            models = {}
            description_files = glob(f"{os.path.join(path, args.model_type)}/*.json")
            for file in description_files:
                name = Path(file).stem.split(".")[0]
                with open(file, "r") as f:
                    content = json.load(f)
                curr = 0
                for profile in content.get("hyperparameter_exploration", []):
                    configs = itertools.product(*list(profile.values()))
                    parameters = list(profile.keys())
                    for config in configs:
                        models[f"{name}-{curr}"] = {
                            k: v for k, v in zip(parameters, config)
                        }
                        curr += 1
            with open(os.path.join(path, "experiment_model_configs.json"), "w") as f:
                f.write(json.dumps(models, indent=4))

    def bagging(self, df, partitions, samples, output):
        print("Starting Bagging partition...")
        indices = np.array(df.index)
        for i in range(partitions):
            print(f"Partition {i} Started")
            np.random.shuffle(indices)
            pandas_df = pd.DataFrame(
                df.loc[indices[:samples]], columns=list(df.columns)
            )
            pandas_df.to_parquet(os.path.join(output, f"df_{i}.parquet"))
            print(f"Partition {i} Done")

    def batching(self, df, partitions, samples, output):
        print("Starting Batching partition...")
        indices = np.array(df.index)
        np.random.shuffle(indices)
        for i in range(partitions):
            if (i + 1) * samples >= len(indices):
                break
            print(f"Partition {i} Started")
            pandas_df = pd.DataFrame(
                df.loc[indices[i * samples : (i + 1) * samples]],
                columns=list(df.columns),
            )
            pandas_df.to_parquet(os.path.join(output, f"df_{i}.parquet"))
            print(f"Partition {i} Done")

    def data_partition(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-p", "--path", help="path to dataframe", type=Path, required=True
        )
        parser.add_argument(
            "-i",
            "--indices",
            help="path to indices split",
            type=Path,
            default="data/split_train_indices.npy",
        )
        parser.add_argument(
            "-n",
            "--n_partitions",
            help="number of data partitions, if 0 partitions are not created",
            type=int,
            default=5,
        )
        parser.add_argument(
            "-s",
            "--partition_samples",
            help="number of sample per partition, if 0 partitions are not created",
            type=int,
            default=200000,
        )
        parser.add_argument(
            "-t",
            "--partition_technique",
            help="use bagging or batching for data points sampling",
            type=str,
            choices=["bagging", "batching"],
            default="bagging",
        )

        parser.add_argument(
            "-e", "--test_partition", help="test partition", action="store_true"
        )
        parser.add_argument(
            "-o", "--output_dir", help="path to output folder", type=Path, required=True
        )
        args = parser.parse_args(sys.argv[2:])

        print("Loading Dask DataFrame...")
        df = ddf.read_parquet(args.path, index="index", calculate_divisions=True)

        print("Loading Data Indices...")
        indices = np.load(args.indices)

        print("Filter DataFrame by indices...")
        df = df.loc[indices]

        if args.test_partition:
            print("Test Partition Started")
            pandas_df = pd.DataFrame(
                df.loc[indices[: args.partition_samples]],
                columns=list(df.columns),
            )
            pandas_df.to_parquet(os.path.join(args.output_dir, f"test.parquet"))
            print(f"Test Partition Done")

        else:
            print("Remove rows with NaN values...")
            df = df.loc[
                ~(
                    np.isnan(df["-1x_raw"])
                    | np.isnan(df["1x_raw"])
                    | np.isnan(df["-1y_raw"])
                    | np.isnan(df["1y_raw"])
                    | np.isnan(df["-1z_raw"])
                    | np.isnan(df["1z_raw"])
                    | np.isnan(df["-2z_raw"])
                    | np.isnan(df["2z_raw"])
                    | np.isnan(df["-3z_raw"])
                    | np.isnan(df["3z_raw"])
                )
            ]
            getattr(self, args.partition_technique)(
                df, args.n_partitions, args.partition_samples, args.output_dir
            )


if __name__ == "__main__":
    TrainingSetup()
