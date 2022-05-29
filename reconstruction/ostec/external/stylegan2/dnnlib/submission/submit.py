# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Submit a function to be run either locally or in a computing cluster."""

import copy
import inspect
import os
import pathlib
import pickle
import platform
import pprint
import re
import shutil
import sys
import time
import traceback

from enum import Enum

from .. import util


class SubmitTarget(Enum):
    """The target where the function should be run.

    LOCAL: Run it locally.
    """
    LOCAL = 1


class PathType(Enum):
    """Determines in which format should a path be formatted.

    WINDOWS: Format with Windows style.
    LINUX: Format with Linux/Posix style.
    AUTO: Use current OS type to select either WINDOWS or LINUX.
    """
    WINDOWS = 1
    LINUX = 2
    AUTO = 3


class PlatformExtras:
    """A mixed bag of values used by dnnlib heuristics.

    Attributes:

        data_reader_buffer_size: Used by DataReader to size internal shared memory buffers.
        data_reader_process_count: Number of worker processes to spawn (zero for single thread operation)
    """
    def __init__(self):
        self.data_reader_buffer_size = 1<<30    # 1 GB
        self.data_reader_process_count = 0      # single threaded default


_user_name_override = None

class SubmitConfig(util.EasyDict):
    """Strongly typed config dict needed to submit runs.

    Attributes:
        run_dir_root: Path to the run dir root. Can be optionally templated with tags. Needs to always be run through get_path_from_template.
        run_desc: Description of the run. Will be used in the run dir and task name.
        run_dir_ignore: List of file patterns used to ignore files when copying files to the run dir.
        run_dir_extra_files: List of (abs_path, rel_path) tuples of file paths. rel_path root will be the src directory inside the run dir.
        submit_target: Submit target enum value. Used to select where the run is actually launched.
        num_gpus: Number of GPUs used/requested for the run.
        print_info: Whether to print debug information when submitting.
        local.do_not_copy_source_files: Do not copy source files from the working directory to the run dir.
        run_id: Automatically populated value during submit.
        run_name: Automatically populated value during submit.
        run_dir: Automatically populated value during submit.
        run_func_name: Automatically populated value during submit.
        run_func_kwargs: Automatically populated value during submit.
        user_name: Automatically populated value during submit. Can be set by the user which will then override the automatic value.
        task_name: Automatically populated value during submit.
        host_name: Automatically populated value during submit.
        platform_extras: Automatically populated values during submit.  Used by various dnnlib libraries such as the DataReader class.
    """

    def __init__(self):
        super().__init__()

        # run (set these)
        self.run_dir_root = ""  # should always be passed through get_path_from_template
        self.run_desc = ""
        self.run_dir_ignore = ["__pycache__", "*.pyproj", "*.sln", "*.suo", ".cache", ".idea", ".vs", ".vscode", "_cudacache"]
        self.run_dir_extra_files = []

        # submit (set these)
        self.submit_target = SubmitTarget.LOCAL
        self.num_gpus = 1
        self.print_info = False
        self.nvprof = False
        self.local = external.stylegan2.dnnlib.submission.internal.local.TargetOptions()
        self.datasets = []

        # (automatically populated)
        self.run_id = None
        self.run_name = None
        self.run_dir = None
        self.run_func_name = None
        self.run_func_kwargs = None
        self.user_name = None
        self.task_name = None
        self.host_name = "localhost"
        self.platform_extras = PlatformExtras()


def get_path_from_template(path_template: str, path_type: PathType = PathType.AUTO) -> str:
    """Replace tags in the given path template and return either Windows or Linux formatted path."""
    # automatically select path type depending on running OS
    if path_type == PathType.AUTO:
        if platform.system() == "Windows":
            path_type = PathType.WINDOWS
        elif platform.system() == "Linux":
            path_type = PathType.LINUX
        else:
            raise RuntimeError("Unknown platform")

    path_template = path_template.replace("<USERNAME>", get_user_name())

    # return correctly formatted path
    if path_type == PathType.WINDOWS:
        return str(pathlib.PureWindowsPath(path_template))
    elif path_type == PathType.LINUX:
        return str(pathlib.PurePosixPath(path_template))
    else:
        raise RuntimeError("Unknown platform")


def get_template_from_path(path: str) -> str:
    """Convert a normal path back to its template representation."""
    path = path.replace("\\", "/")
    return path


def convert_path(path: str, path_type: PathType = PathType.AUTO) -> str:
    """Convert a normal path to template and the convert it back to a normal path with given path type."""
    path_template = get_template_from_path(path)
    path = get_path_from_template(path_template, path_type)
    return path


def set_user_name_override(name: str) -> None:
    """Set the global username override value."""
    global _user_name_override
    _user_name_override = name


def get_user_name():
    """Get the current user name."""
    if _user_name_override is not None:
        return _user_name_override
    elif platform.system() == "Windows":
        return os.getlogin()
    elif platform.system() == "Linux":
        try:
            import pwd
            return pwd.getpwuid(os.geteuid()).pw_name
        except:
            return "unknown"
    else:
        raise RuntimeError("Unknown platform")


def make_run_dir_path(*paths):
    """Make a path/filename that resides under the current submit run_dir.

    Args:
        *paths: Path components to be passed to os.path.join

    Returns:
        A file/dirname rooted at submit_config.run_dir.  If there's no
        submit_config or run_dir, the base directory is the current
        working directory.

    E.g., `os.path.join(dnnlib.submit_config.run_dir, "output.txt"))`
    """
    import dnnlib
    if (dnnlib.submit_config is None) or (dnnlib.submit_config.run_dir is None):
        return os.path.join(os.getcwd(), *paths)
    return os.path.join(dnnlib.submit_config.run_dir, *paths)


def _create_run_dir_local(submit_config: SubmitConfig) -> str:
    """Create a new run dir with increasing ID number at the start."""
    run_dir_root = get_path_from_template(submit_config.run_dir_root, PathType.AUTO)

    if not os.path.exists(run_dir_root):
        os.makedirs(run_dir_root)

    submit_config.run_id = _get_next_run_id_local(run_dir_root)
    submit_config.run_name = "{0:05d}-{1}".format(submit_config.run_id, submit_config.run_desc)
    run_dir = os.path.join(run_dir_root, submit_config.run_name)

    if os.path.exists(run_dir):
        raise RuntimeError("The run dir already exists! ({0})".format(run_dir))

    os.makedirs(run_dir)

    return run_dir


def _get_next_run_id_local(run_dir_root: str) -> int:
    """Reads all directory names in a given directory (non-recursive) and returns the next (increasing) run id. Assumes IDs are numbers at the start of the directory names."""
    dir_names = [d for d in os.listdir(run_dir_root) if os.path.isdir(os.path.join(run_dir_root, d))]
    r = re.compile("^\\d+")  # match one or more digits at the start of the string
    run_id = 0

    for dir_name in dir_names:
        m = r.match(dir_name)

        if m is not None:
            i = int(m.group())
            run_id = max(run_id, i + 1)

    return run_id


def _populate_run_dir(submit_config: SubmitConfig, run_dir: str) -> None:
    """Copy all necessary files into the run dir. Assumes that the dir exists, is local, and is writable."""
    pickle.dump(submit_config, open(os.path.join(run_dir, "submit_config.pkl"), "wb"))
    with open(os.path.join(run_dir, "submit_config.txt"), "w") as f:
        pprint.pprint(submit_config, stream=f, indent=4, width=200, compact=False)

    if (submit_config.submit_target == SubmitTarget.LOCAL) and submit_config.local.do_not_copy_source_files:
        return

    files = []

    run_func_module_dir_path = util.get_module_dir_by_obj_name(submit_config.run_func_name)
    assert '.' in submit_config.run_func_name
    for _idx in range(submit_config.run_func_name.count('.') - 1):
        run_func_module_dir_path = os.path.dirname(run_func_module_dir_path)
    files += util.list_dir_recursively_with_ignore(run_func_module_dir_path, ignores=submit_config.run_dir_ignore, add_base_to_relative=False)

    dnnlib_module_dir_path = util.get_module_dir_by_obj_name("dnnlib")
    files += util.list_dir_recursively_with_ignore(dnnlib_module_dir_path, ignores=submit_config.run_dir_ignore, add_base_to_relative=True)

    files += submit_config.run_dir_extra_files

    files = [(f[0], os.path.join(run_dir, "src", f[1])) for f in files]
    files += [(os.path.join(dnnlib_module_dir_path, "submission", "internal", "run.py"), os.path.join(run_dir, "run.py"))]

    util.copy_files_and_create_dirs(files)



def run_wrapper(submit_config: SubmitConfig) -> None:
    """Wrap the actual run function call for handling logging, exceptions, typing, etc."""
    is_local = submit_config.submit_target == SubmitTarget.LOCAL

    # when running locally, redirect stderr to stdout, log stdout to a file, and force flushing
    if is_local:
        logger = util.Logger(file_name=os.path.join(submit_config.run_dir, "log.txt"), file_mode="w", should_flush=True)
    else:  # when running in a cluster, redirect stderr to stdout, and just force flushing (log writing is handled by run.sh)
        logger = util.Logger(file_name=None, should_flush=True)

    import dnnlib
    dnnlib.submit_config = submit_config

    exit_with_errcode = False
    try:
        print("dnnlib: Running {0}() on {1}...".format(submit_config.run_func_name, submit_config.host_name))
        start_time = time.time()

        run_func_obj = util.get_obj_by_name(submit_config.run_func_name)
        assert callable(run_func_obj)
        sig = inspect.signature(run_func_obj)
        if 'submit_config' in sig.parameters:
            run_func_obj(submit_config=submit_config, **submit_config.run_func_kwargs)
        else:
            run_func_obj(**submit_config.run_func_kwargs)

        print("dnnlib: Finished {0}() in {1}.".format(submit_config.run_func_name, util.format_time(time.time() - start_time)))
    except:
        if is_local:
            raise
        else:
            traceback.print_exc()

            log_src = os.path.join(submit_config.run_dir, "log.txt")
            log_dst = os.path.join(get_path_from_template(submit_config.run_dir_root), "{0}-error.txt".format(submit_config.run_name))
            shutil.copyfile(log_src, log_dst)

            # Defer sys.exit(1) to happen after we close the logs and create a _finished.txt
            exit_with_errcode = True
    finally:
        open(os.path.join(submit_config.run_dir, "_finished.txt"), "w").close()

    dnnlib.RunContext.get().close()
    dnnlib.submit_config = None
    logger.close()

    # If we hit an error, get out of the script now and signal the error
    # to whatever process that started this script.
    if exit_with_errcode:
        sys.exit(1)

    return submit_config


def submit_run(submit_config: SubmitConfig, run_func_name: str, **run_func_kwargs) -> None:
    """Create a run dir, gather files related to the run, copy files to the run dir, and launch the run in appropriate place."""
    submit_config = copy.deepcopy(submit_config)

    submit_target = submit_config.submit_target
    farm = None
    if submit_target == SubmitTarget.LOCAL:
        farm = external.stylegan2.dnnlib.submission.internal.local.Target()
    assert farm is not None # unknown target

    # Disallow submitting jobs with zero num_gpus.
    if (submit_config.num_gpus is None) or (submit_config.num_gpus == 0):
        raise RuntimeError("submit_config.num_gpus must be set to a non-zero value")

    if submit_config.user_name is None:
        submit_config.user_name = get_user_name()

    submit_config.run_func_name = run_func_name
    submit_config.run_func_kwargs = run_func_kwargs

    #--------------------------------------------------------------------
    # Prepare submission by populating the run dir
    #--------------------------------------------------------------------
    host_run_dir = _create_run_dir_local(submit_config)

    submit_config.task_name = "{0}-{1:05d}-{2}".format(submit_config.user_name, submit_config.run_id, submit_config.run_desc)
    docker_valid_name_regex = "^[a-zA-Z0-9][a-zA-Z0-9_.-]+$"
    if not re.match(docker_valid_name_regex, submit_config.task_name):
        raise RuntimeError("Invalid task name.  Probable reason: unacceptable characters in your submit_config.run_desc.  Task name must be accepted by the following regex: " + docker_valid_name_regex + ", got " + submit_config.task_name)

    # Farm specific preparations for a submit
    farm.finalize_submit_config(submit_config, host_run_dir)
    _populate_run_dir(submit_config, host_run_dir)
    return farm.submit(submit_config, host_run_dir)
