import argparse
import csv
import json
import os
import traceback
from glob import glob
from pathlib import Path
from multiprocessing import Process, Value
from time import perf_counter
from tqdm import tqdm

import numpy as np
import pandas as pd

from wrapper_models import Regressor, Cluster


def instantiate_model(wrapper_class, model_file, hyperparams):
    with open(model_file, "r") as f:
        model_specs = json.load(f)
    preprocessing_modules = []
    preprocessing_params = []
    for pre in model_specs.get("preprocessing", []):
        preprocessing_modules.append(pre["module"])
    for param in list(hyperparams.keys()):
        if "_pre_" in param:
            _, _, index, p = param.split("_")
            index = int(index)
            while len(preprocessing_params) <= index:
                preprocessing_params.append({})
            preprocessing_params[index][p] = hyperparams.pop(param)
        if "default" in hyperparams:
            hyperparams.pop("default")

    return wrapper_class(
        model_specs["module"],
        model_parameters=hyperparams,
        preprocessing=preprocessing_modules,
        pre_parameters=preprocessing_params,
    )


def train_model(
    model_file,
    wrapper_class,
    parameters,
    problem,
    features,
    dataframe_file,
    all_train_dataframes,
    test_dataframe,
    success_flag,
    *results,
):
    model = instantiate_model(
        wrapper_class=wrapper_class, model_file=model_file, hyperparams=parameters
    )

    # Training and Training metrics
    pandas_df = pd.read_parquet(dataframe_file)

    model.fit(pandas_df[features].values, pandas_df[problem].values)
    metrics = []

    y_train = model.predict(pandas_df[features].values)
    y_hat = pandas_df[problem].values

    metrics.extend(model.compute_metrics(y_train, y_hat))
    del pandas_df
    del y_train

    # Getting all training metrics
    y_pred = []
    y_hat = []
    for file in all_train_dataframes:
        pandas_df = pd.read_parquet(file)
        y_pred.append(model.predict(pandas_df[features].values))
        y_hat.append(pandas_df[problem].values)

    metrics.extend(
        model.compute_metrics(np.asarray(y_pred).flatten(), np.asarray(y_hat).flatten())
    )

    # Getting test metrics
    pandas_df = pd.read_parquet(test_dataframe)
    y_pred = model.predict(pandas_df[features].values)
    y_hat = pandas_df[problem].values
    metrics.extend(model.compute_metrics(y_pred, y_hat))

    for i, r in enumerate(results):
        r.value = metrics[i]
    success_flag.value = 1


def test_model_creation(model_dir, model_config, wrapper_class, stop):
    failed = {}
    for model_instance, parameters in tqdm(model_config.items(), desc="model"):
        try:
            model_name = model_instance.split("-")[0]
            model_file = os.path.join(model_dir, f"{model_name}.json")
            instantiate_model(
                wrapper_class=wrapper_class,
                model_file=model_file,
                hyperparams=parameters,
            )
        except Exception as e:
            if stop:
                print(model_instance)
                print(traceback.format_exc())
                break
            failed[model_instance] = f"{e.__class__.__name__}: {str(e)}"
    if not stop:
        print(
            f"Successful Instantiations: {(len(model_config) - len(failed))}/{len(model_config)}"
        )
        print(failed)


def train_models(
    model_dir,
    model_config,
    data_features,
    wrapper_class,
    csv_columns,
    training_path,
    base_dataset,
    number_datasets,
):
    datasets_all = sorted(glob(os.path.join(training_path, "df*.parquet")))[
        :number_datasets
    ]
    datasets = datasets_all
    if base_dataset is not None:
        datasets = [datasets[base_dataset]]

    results_size = 3 * len(wrapper_class.metrics)
    results = [Value("d", 0.0) for _ in range(results_size)]
    success = Value("i", 0)
    for dataset in tqdm(datasets, desc="dataset", position=0):
        for problem, feature_sets in tqdm(
            data_features.items(), desc="problem", position=1
        ):
            with open(
                os.path.join(
                    training_path, f"{Path(dataset).stem.split('.')[0]}-{problem}.csv"
                ),
                "w",
            ) as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(csv_columns)
                for set_name, features in tqdm(
                    feature_sets.items(), desc="feature_set", position=2
                ):
                    for model, parameters in tqdm(
                        model_config.items(), desc="model", position=3
                    ):
                        model_name = model.split("-")[0]
                        model_file = os.path.join(model_dir, f"{model_name}.json")
                        for r in results:
                            r.value = 0.0
                        success.value = 0
                        p = Process(
                            target=train_model,
                            args=(
                                model_file,
                                wrapper_class,
                                parameters,
                                problem,
                                features,
                                dataset,
                                datasets_all,
                                os.path.join(training_path, "test.parquet"),
                                success,
                                *results,
                            ),
                        )
                        start = perf_counter()
                        p.start()
                        p.join()
                        time = perf_counter() - start
                        row = [
                            model,
                            set_name,
                            *[r.value for r in results],
                            success.value,
                            time,
                        ]
                        csv_writer.writerow(row)


class TrainingRun:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("command", choices=["cluster", "regression"])
        parser.add_argument(
            "-p",
            "--training_path",
            help="path to training datasets",
            type=Path,
            default="data_bag",
        )

        parser.add_argument(
            "-b",
            "--base_dataset",
            help="dataset used for training, if not specified all loops are executed",
            type=int,
            default=None,
        )

        parser.add_argument(
            "-n",
            "--number_datasets",
            help="number of datasets to use",
            type=int,
            default=5,
        )

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

        args = parser.parse_args()
        if not hasattr(self, args.command):
            print("Invalid command")
            parser.print_help()
            exit(1)

        getattr(self, args.command)(args)

    def cluster(self, args):
        wrapper = Cluster

        metrics = [
            f"{split}_{metric}"
            for split in ["train", "test"]
            for metric in wrapper.metrics.keys()
        ]
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
            train_models(
                model_dir,
                model_config,
                data_features,
                wrapper,
                csv_columns,
                args.training_path,
                args.base_dataset,
                args.numbers_dataset,
            )

    def regression(self, args):
        wrapper = Regressor

        metrics = [
            f"{split}_{metric}"
            for split in ["train", "test"]
            for metric in wrapper.metrics.keys()
        ]
        csv_columns = ["model", "feature_set", *metrics, "success", "time"]
        with open(
            os.path.join(args.model_config, "experiment_model_configs.json"), "r"
        ) as f:
            model_config = json.load(f)
        with open(
            os.path.join(args.model_config, "regression_features.json"), "r"
        ) as f:
            data_features = json.load(f)
        model_dir = os.path.join(args.model_config, "pre-defined")
        if args.instantiation_test:
            test_model_creation(
                model_dir, model_config, wrapper, args.stop_instantiation
            )
        else:
            train_models(
                model_dir,
                model_config,
                data_features,
                wrapper,
                csv_columns,
                args.training_path,
                args.base_dataset,
                args.numbers_dataset,
            )


if __name__ == "__main__":
    TrainingRun()
