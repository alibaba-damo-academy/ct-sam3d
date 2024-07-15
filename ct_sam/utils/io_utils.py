import importlib
import os
import sys


def load_module_from_file(pyfile: str):
    """
    load module from .py file

    :param pyfile: path to the module file
    :return: module
    """

    dirname = os.path.dirname(pyfile)
    basename = os.path.basename(pyfile)
    module_name, _ = os.path.splitext(basename)

    need_reload = module_name in sys.modules

    # to avoid duplicate module name with existing modules, add the specified path first
    os.sys.path.insert(0, dirname)
    module = importlib.import_module(module_name)
    if need_reload:
        importlib.reload(module)
    os.sys.path.pop(0)

    return module