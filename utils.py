import json
import os
import traceback

from tqdm import tqdm


def split_module_class(string):
    split = string.split(".")
    module_name = ".".join(split[:-1])
    class_name = split[-1]
    return module_name, class_name


def instantiate_model(wrapper_class, model_file, hyperparams):
    with open(model_file, "r") as f:
        model_specs = json.load(f)
    preprocessing_modules = []
    preprocessing_params = []
    for pre in model_specs.get("preprocessing", []):
        preprocessing_modules.append(pre["module"])
    while len(preprocessing_params) <= len(preprocessing_modules):
        preprocessing_params.append({})
    for param in list(hyperparams.keys()):
        if "_pre_" in param:
            _, _, index, p = param.split("_")
            index = int(index)
            preprocessing_params[index][p] = hyperparams.pop(param)
        if "default" in hyperparams:
            hyperparams.pop("default")

    return wrapper_class(
        model_specs["module"],
        model_parameters=hyperparams,
        preprocessing=preprocessing_modules,
        pre_parameters=preprocessing_params,
    )


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
