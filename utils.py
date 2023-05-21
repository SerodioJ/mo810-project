def split_module_class(string):
    split = string.split(".")
    module_name = ".".join(split[:-1])
    class_name = split[-1]
    return module_name, class_name
