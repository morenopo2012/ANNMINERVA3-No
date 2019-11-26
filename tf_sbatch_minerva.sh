#!/bin/bash

SCRIPTKEY=`date +%Y%m%d-%H%M%S` #`date +%s`

NGPU=1
NODES=gpu2
#NODES=gpu3
NODES=gpu4

# show what we will do...
cat << EOF
sbatch --gres=gpu:${NGPU} --nodelist=${NODES} -A minervag \
 -p gpu run_estimator_hadmult_simple.sh
EOF

# do the thing, etc.
sbatch --gres=gpu:${NGPU} --nodelist=${NODES} -A minervag \
 -p gpu -o logs/${SCRIPTKEY}.log run_estimator_hadmult_simple.sh

