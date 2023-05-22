import argparse
import csv
import json
import os
from pathlib import Path
from multiprocessing import Process, Value
from time import perf_counter
from tqdm import tqdm

import numpy as np
import dask.dataframe as ddf

from wrapper_models import Cluster
from utils import instantiate_model, test_model_creation


def cluster_data_lazy(
    model_file,
    wrapper_class,
    parameters,
    features,
    dask_dataframe,
    success_flag,
    *results,
):
    model = instantiate_model(
        wrapper_class=wrapper_class, model_file=model_file, hyperparams=parameters
    )

    # Training and Training metrics
    pandas_df = ddf.read_parquet(
        dask_dataframe, index="index", calculate_divisions=True
    )
    X = pandas_df[features].values
    labels_pred = model.fit_predict(X)
    labels_true = np.load("data/F3_train_labels.npy")
    labels_true = labels_true.flatten()

    metrics = model.compute_metrics(labels_pred, labels_true)
    for i, r in enumerate(results):
        r.value = metrics[i]
    success_flag.value = 1


def cluster_inline(
    model_file,
    wrapper_class,
    parameters,
    dask_data,
    labels_true,
    shape,
    success_flag,
    *results,
):
    model = instantiate_model(
        wrapper_class=wrapper_class, model_file=model_file, hyperparams=parameters
    )
    inlines = shape[0]
    inline_size = shape[1] * shape[2]

    inlines_metrics = np.zeros((len(model.metrics), shape[0]))

    for i in range(inlines):
        labels_pred = model.fit_predict(
            dask_data[i * inline_size : (i + 1) * inline_size]
        )
        metrics = model.compute_metrics(
            labels_pred, labels_true[i * inline_size : (i + 1) * inline_size]
        )
        for j, metric in enumerate(metrics):
            inlines_metrics[j][i] = metric

    metrics = np.mean(inlines_metrics, axis=1)

    for i, r in enumerate(results):
        r.value = metrics[i]
    success_flag.value = 1


def cluster_data(
    model_file,
    wrapper_class,
    parameters,
    dask_data,
    labels_true,
    success_flag,
    *results,
):
    model = instantiate_model(
        wrapper_class=wrapper_class, model_file=model_file, hyperparams=parameters
    )

    # Training and Training metrics
    labels_pred = model.fit_predict(dask_data)

    metrics = model.compute_metrics(labels_pred, labels_true)
    for i, r in enumerate(results):
        r.value = metrics[i]
    success_flag.value = 1


def cluster_eval(
    model_dir,
    model_config,
    data_features,
    wrapper_class,
    csv_columns,
    dask_dataframe,
    process,
    inline,
):
    results_size = len(wrapper_class.metrics)
    results = [Value("d", 0.0) for _ in range(results_size)]
    success = Value("i", 0)
    for problem, feature_sets in tqdm(
        data_features.items(), desc="problem", position=0
    ):
        with open(
            os.path.join(model_dir, f"{problem}.csv"),
            "w",
        ) as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(csv_columns)
            for set_name, features in tqdm(
                feature_sets.items(), desc="feature_set", position=1
            ):
                if not process:
                    dask_data = ddf.read_parquet(
                        dask_dataframe, index="index", calculate_divisions=True
                    )
                    dask_data = dask_data[features].values.compute()
                    labels_true = np.load("data/F3_train_labels.npy")
                    shape = labels_true.shape
                    labels_true = labels_true.flatten()

                for model, parameters in tqdm(
                    model_config.items(), desc="model", position=2
                ):
                    model_name = model.split("-")[0]
                    model_file = os.path.join(model_dir, f"{model_name}.json")
                    for r in results:
                        r.value = 0.0
                    success.value = 0
                    if process:
                        p = Process(
                            target=cluster_data_lazy,
                            args=(
                                model_file,
                                wrapper_class,
                                parameters,
                                features,
                                dask_dataframe,
                                success,
                                *results,
                            ),
                        )
                        start = perf_counter()
                        p.start()
                        p.join()
                        time = perf_counter() - start
                    elif inline:
                        start = perf_counter()
                        cluster_inline(
                            model_file,
                            wrapper_class,
                            parameters,
                            dask_data,
                            labels_true,
                            shape,
                            success,
                            *results,
                        )
                        time = perf_counter() - start
                    else:
                        start = perf_counter()
                        cluster_data(
                            model_file,
                            wrapper_class,
                            parameters,
                            dask_data,
                            labels_true,
                            success,
                            *results,
                        )
                        time = perf_counter() - start
                    row = [
                        model,
                        set_name,
                        *[r.value for r in results],
                        success.value,
                        time,
                    ]
                    csv_writer.writerow(row)


class ClusteringEval:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("command", choices=["cluster"])

        parser.add_argument(
            "-d",
            "--dask_dataframe",
            help="path to dask_dataframe",
            type=Path,
            default="data/dataframe.parquet",
        )

        parser.add_argument(
            "-m",
            "--model_config",
            help="path to model config files (features and hyperparameters)",
            type=Path,
            required=True,
        )

        parser.add_argument(
            "-i",
            "--instantiation_test",
            help="instatiates all models listed in model_config file to check if the object is created correctly",
            action="store_true",
        )

        parser.add_argument(
            "-s",
            "--stop_instantiation",
            help="stops instantiation at first fail and shows traceback",
            action="store_true",
        )

        parser.add_argument(
            "-p",
            "--process",
            help="runs training and prediction on diferent process, loading data from disk on every computation",
            action="store_true",
        )

        parser.add_argument(
            "-l",
            "--inline",
            help="clustering is done per inline",
            action="store_true",
        )

        args = parser.parse_args()
        if not hasattr(self, args.command):
            print("Invalid command")
            parser.print_help()
            exit(1)

        getattr(self, args.command)(args)

    def cluster(self, args):
        wrapper = Cluster

        metrics = [metric for metric in wrapper.metrics.keys()]
        csv_columns = ["model", "feature_set", *metrics, "sucess", "time"]

        with open(
            os.path.join(args.model_config, "experiment_model_configs.json"), "r"
        ) as f:
            model_config = json.load(f)
        with open(
            os.path.join(args.model_config, "clustering_features.json"), "r"
        ) as f:
            data_features = json.load(f)
        model_dir = os.path.join(args.model_config, "pre-defined")
        if args.instantiation_test:
            test_model_creation(
                model_dir, model_config, wrapper, args.stop_instantiation
            )
        else:
            cluster_eval(
                model_dir,
                model_config,
                data_features,
                wrapper,
                csv_columns,
                args.dask_dataframe,
                args.process,
                args.inline,
            )


if __name__ == "__main__":
    ClusteringEval()
