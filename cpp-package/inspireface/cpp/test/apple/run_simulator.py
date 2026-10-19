#!/usr/bin/env python3
"""Run an iOS simulator contract binary on an already booted simulator."""
import os
import subprocess
import sys

env = dict(os.environ)
if 'LLVM_PROFILE_FILE' in env:
    env['SIMCTL_CHILD_LLVM_PROFILE_FILE'] = env['LLVM_PROFILE_FILE']
# A missing simulator/service is a test failure, never silently counted as a pass.
result = subprocess.run(['xcrun', 'simctl', 'spawn', os.environ.get('ISF_APPLE_SIMULATOR', 'booted'), *sys.argv[1:]], env=env)
sys.exit(result.returncode)
