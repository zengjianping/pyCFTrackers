#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH

python examples/demo.py \
    --tracker_type ECO-HC \
    --video_name datas/sequences/Crossing/
