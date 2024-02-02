#!/bin/bash

ALGOS="DSST ECO-HC"
VIDEOS1=`ls datas/videos/*.mp4`
VIDEOS2=`ls datas/videos/*.avi`
VIDEOS="$VIDEOS1 $VIDEOS2"

for i in $(seq 0 1)
do
    EXPAND="--expand_roi"
    if [ $i == 0 ]; then
        EXPAND=""
    fi
    for ALGO in $ALGOS;
    do
        echo "processing $ALGO..."
        for VIDEO in $VIDEOS;
        do
            echo "processing $VIDEO..."
            cmd="python examples/demo.py --tracker_type $ALGO --video_name $VIDEO --save_result $EXPAND"
            echo $cmd
            eval $cmd
        done
    done
done

