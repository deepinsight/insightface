# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

class TargetOptions():
    def __init__(self):
        self.do_not_copy_source_files = False

class Target():
    def __init__(self):
        pass

    def finalize_submit_config(self, submit_config, host_run_dir):
        print ('Local submit ', end='', flush=True)
        submit_config.run_dir = host_run_dir

    def submit(self, submit_config, host_run_dir):
        from ..submit import run_wrapper, convert_path
        print('- run_dir: %s' % convert_path(submit_config.run_dir), flush=True)
        return run_wrapper(submit_config)
