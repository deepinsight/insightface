#!/usr/bin/env bash

DEVKIT="/raid5data/dplearn/megaface/devkit/experiments"
ALGO="r100ii" #ms1mv2
ROOT=$(dirname `which $0`)
echo $ROOT
python -u gen_megaface.py --gpu 0 --algo "$ALGO" --model '../../models2/model-r100-ii/model,0'
python -u remove_noises.py --algo "$ALGO"

cd "$DEVKIT"
LD_LIBRARY_PATH="/usr/local/lib64:$LD_LIBRARY_PATH" python -u run_experiment.py "$ROOT/feature_out_clean/megaface" "$ROOT/feature_out_clean/facescrub" _"$ALGO".bin ../../mx_results/ -s 1000000 -p ../templatelists/facescrub_features_list.json
cd -

