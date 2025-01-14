#!/usr/bin/env bash
# debug_xvfb.sh
#
# This script starts your Python program under xvfb-run
# with debugpy, waiting for VSCode to attach.

# You can add any custom flags after python ...
xvfb-run -a python -m debugpy --listen 5678 --wait-for-client \
    -u action_extractor/megapose/imitate_trajectory_with_megapose.py "$@"