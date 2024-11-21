# meta.py
#
# Add meta features to functions of a module
#
# Copyright (c) 2024 CWordTM Project 
# Author: Johnny Cheng <drjohnnycheng@gmail.com>
#
# Updated: 15 June 2024 (0.6.4)
#
# URL: https://github.com/drjohnnycheng/cwordtm.git
# For license information, see LICENSE.TXT

import ast
import inspect
from functools import wraps
import pkgutil
from importlib import import_module
import types
import time


def get_module_info(detailed=False):
    """Gets the information of the module 'cwordtm'.

    :param detailed: The flag indicating whether only function signature or
        detailed source code is shown, default to False
    :type detailed: bool, optional
    :return: The information of the module 'cwordtm'
    :rtype: str
    """

    mod_name = __name__.split('.')[0]   # Current module
    module = import_module(mod_name)
    func1 = "@validate_params"
    func2 = "def cosine_similarity"

    i = 0
    mod_info = "The member information of the module '" + mod_name + "'\n"
    for _, submodname, ispkg in pkgutil.iter_modules(module.__path__):
        if not ispkg:
            i += 1
            mod_info += "%d. Submodule %s:" %(i, submodname) + "\n"
            submod = import_module(".."+submodname, mod_name+"."+submodname)
            for name, member in inspect.getmembers(submod):
                if (inspect.isclass(member) and name in ['LDA', 'NMF', 'BTM']) or \
                   (inspect.isfunction(member) and name != 'files' and
                    not name in func1 and not name in func2):
                    if detailed:
                        mod_info += "{}\n".format(inspect.getsource(member))
                    else:
                        mod_info += "   {} {}\n".format(name, inspect.signature(member))

    return mod_info


def get_submodule_info(submodname, detailed=False):
    """Gets the information of the prescribed submodule of the module 'cwordtm'.

    :param submodname: The name of the prescribed submodule, default to None
    :type submodname: str
    :param detailed: The flag indicating whether only function signature or
        detailed source code is shown, default to False
    :type detailed: bool, optional
    :return: The information of the prescribed submodule
    :rtype: str
    """

    mod_name = __name__.split('.')[0]   # Current module
    module = import_module(mod_name)

    mod_info = ""
    for _, submod, ispkg in pkgutil.iter_modules(module.__path__):
        if not ispkg and submod == submodname:
            submod_obj = import_module(".."+submodname, mod_name+"."+submodname)
            for name, member in inspect.getmembers(submod_obj):
                if (inspect.isclass(member) and name in ['LDA', 'NMF', 'BTM']) or \
                   (inspect.isfunction(member) and name != 'files'):
                    if detailed:
                        mod_info += "{}\n".format(inspect.getsource(member))
                    else:
                        mod_info += "   {} {}\n".format(name, inspect.signature(member))

    if len(mod_info) == 0:
        mod_info = "The submodule '" + submodname + "' cannot be found!"
    else:
        mod_info = "The function(s) of the submodule '" + mod_name + "." + submodname + "':\n\n" + mod_info

    return mod_info


def get_function(mod_name, submodules, func_name):
    """Gets the object of the function 'func_name' if it belongs
    to one of 'submodules' of the current top-level module.

    :param mod_name: The name of the source top-level module, default to None
    :type mod_name: str
    :param submodules: The list of names of the sub-modules of the top-level module
    :type submodules: list
    :param func_name: The name of the fuunction to be looked for
    :type func_name: str
    :return: The object of the target function, if any, otherwise None
    :rtype: function
    """

    for submod in submodules:
        mod_obj = import_module(mod_name + "." + submod)
        if hasattr(mod_obj, func_name):
            return getattr(mod_obj, func_name), submod

    return None, None


def addin(func):
    """Adds additional features (showing timing information and source code)
    to a function at runtime. This adds two parameters ('timing' & 'code') to
    function 'func' at runtime. 'timing' is a flag indicating whether
    the execution time of the function is shown, and it is default to False.
    'code' is an indicator determining if the source code of the function
    'func' is shown and/or the function is invoked; '0' indicates the function
    is executed but its source code is not shown, '1' indicates the source code
    of the function is shown after execution, or '2' indicates the source code
    of the function is shown without execution, and it is default to 0.

    :param func: The target function for inserting additiolnal features - 
        timing information and showing code, default to None
    :type func: function
    :return: The wrapper function
    :rtype: function
    """

    try:
        if "code" in inspect.signature(func).parameters:
            return
    except ValueError:
        return

    mod_name = __name__.split('.')[0]
    module = import_module(mod_name)

    # Get Submodules of cwordtm
    submodules = [name for name in dir(module) 
                    if isinstance(getattr(module, name), type(module))]

    exclusion = ["files", "WordCloud", "ngrams"]

    def next_level(func):
        source_code = inspect.getsource(func)
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_call_code = ast.get_source_segment(source_code, node)
                func_name = func_call_code.split('(')[0].split(".")[-1]
                func_obj, submod = get_function(mod_name, submodules, func_name)
                if func_obj is not None and func_name not in exclusion:
                    module_name = inspect.getmodule(func_obj).__name__
                    print(">>", module_name + "." + func_name)
                    print(inspect.getsource(func_obj))

    @wraps(func)
    def wrapper(*args, timing=False, code=0, **kwargs):
        """Wrapper function to add two parameters ('timing' & 'code') to
        function 'func' at runtime.

        :param timing: The flag indicating whether the execution time of the
        function 'func' is shown, default to False
        :type timing: bool, optional
        :param code: The indicator determining if the source code of the function
            'func' is shown and/or the function is invoked; '0' indicates the function
            is executed but its source code is not shown, '1' indicates the source code
            of the function is shown after execution, or '2' indicates the source code
            of the function is shown without execution, default to 0
        :type code: bool, optional
        :return: The execution time (in seconds) of the function 'func'
        :rtype: float
        """

        if code == 0 or code == 1:
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time

            if timing:
                print(f"Finished {func.__name__!r} in {run_time:.4f} secs")

            if code == 1:
                if isinstance(func, types.BuiltinFunctionType):
                    print("\nSource code of the builtin function '%s' cannot be retrieved!" \
                          %func.__name__)
                else:
                    print("\n" + inspect.getsource(func))
                    next_level(func)
 
            return value
        elif code == 2:
            if isinstance(func, types.BuiltinFunctionType):
                print("\nSource code of the funcrion '%s' cannot be retrieved!" \
                      %func.__name__)
            else:
                print("\n" + inspect.getsource(func) + "\n")
                next_level(func)

            return None

    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    param_time = inspect.Parameter("timing",
                                   inspect.Parameter.KEYWORD_ONLY,
                                   default=False)
    param_code = inspect.Parameter("code",
                                   inspect.Parameter.KEYWORD_ONLY,
                                   default=0)

    if 'kwargs' in list(sig.parameters):
        kw_loc = list(sig.parameters).index('kwargs')
        params.insert(kw_loc, param_code)
        params.insert(kw_loc, param_time)
    else:
        params.append(param_time)
        params.append(param_code)

    try:
        wrapper.__signature__ = sig.replace(parameters=params)
    except:
        print(">> Exception:", func.__name__)
        print(">>>", params)

    return wrapper


def addin_all_functions(submod):
    """Applies 'addin' function to all functions of a module at runtime.

    :param submod: The target sub-module of which all the functions are inserted
        additional features, default to None
    :type submod: module
    """

    for name, member in inspect.getmembers(submod):
        if callable(member) and \
           member.__name__ != 'files' and \
           name[0].islower():
            setattr(submod, name, addin(member))


def addin_all(modname='cwordtm'):
    """Applies 'addin' function to all functions of all sub-modules of
    a module at runtime.

    :param modname: The target module of which all the functions are inserted
        additional features, default to 'wordtm'
    :type modname: str, optional
    """

    module = import_module(modname)

    if hasattr(module, "__path__"):
        for _, submodname, ispkg in pkgutil.iter_modules(module.__path__):
            # if not ispkg and submodname != 'meta':
            if not ispkg:
                submod = import_module(".." + submodname, \
                                       modname + "." + submodname)
                addin_all_functions(submod)
    else:
        addin_all_functions(module)
