#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=tf_hadmult

echo "started "`date`" "`date +%s`""

nvidia-smi -L

SING_DIR="/lfstev/e-938/jbonilla/sing_imgs"
SINGLRTY="${SING_DIR}/joaocaldeira-singularity_imgs-master-py3_tf114.simg"

EXE=estimator_hadmult_simple.py
NCLASSES=6
BATCH_SIZE=100
TRAIN_STEPS=235280
VALID_STEPS=20000
let SAVE_STEPS=TRAIN_STEPS/20
NUM_EPOCHS=12
#MODEL_DIR=/data/minerva/JLBRtesthad/tensorflow/models/ResNet18/ # 64 filter
#MODEL_DIR=/data/minerva/JLBRtesthad/tensorflow/models/ResNet18_2/ # 32
#MODEL_DIR=/data/minerva/JLBRtesthad/tensorflow/models/ResNet50/ # 32
#MODEL_DIR=/data/minerva/JLBRtesthad/tensorflow/models/ResNet50_2 # No bottleneck
#MODEL_DIR=/data/minerva/JLBRtesthad/tensorflow/models/ResNet50_3 # 64
#MODEL_DIR=/data/minerva/JLBRtesthad/tensorflow/models/tests
#MODEL_DIR=/data/minerva/JLBRtesthad/tensorflow/models/vtx
#MODEL_DIR=/data/minerva/JLBRtesthad/tensorflow/models/vtx_ResNet50_64_bottleneck # 64
MODEL_DIR=/data/minerva/JLBRtesthad/tensorflow/models/vtx_based_bilinear_2 # row normalization
MODEL_DIR=/data/minerva/JLBRtesthad/tensorflow/models/vtx_based_bilinear_3 # "column" normalization
MODEL=model.ckpt-94112 #"model.ckpt-70584" # model for prediction
DATA_DIR=/lfstev/e-938/jbonilla/hdf5
TRAIN_FILE=${DATA_DIR}/hadmultkineimgs_127x94_me1Nmc.hdf5 # Can be multiple files, separated by space (not yet)
EVAL_FILE=${DATA_DIR}/hadmultkineimgs_127x94_me1Fmc.hdf5
TARGET=hadro_data/n_hadmultmeas
#TARGET=vtx_data/planecodes

if [ ! -d "$MODEL_DIR" ]
then
  mkdir $MODEL_DIR
fi

ARGS="--batch-size ${BATCH_SIZE}"
ARGS+=" --nclasses ${NCLASSES}"
ARGS+=" --train-steps ${TRAIN_STEPS}"
ARGS+=" --valid-steps ${VALID_STEPS}"
ARGS+=" --save-steps ${SAVE_STEPS}"
ARGS+=" --num-epochs ${NUM_EPOCHS}"
ARGS+=" --train-file ${TRAIN_FILE}"
ARGS+=" --eval-file ${EVAL_FILE}"
ARGS+=" --target_field ${TARGET}"
ARGS+=" --model-dir ${MODEL_DIR}"
ARGS+=" --model ${MODEL}"
ARGS+=" --do-train"
#ARGS+=" --do-test"

cat << EOF
singularity exec --nv $SINGLRTY python3 $EXE $ARGS
EOF

singularity exec --nv $SINGLRTY python3  $EXE $ARGS
#-m cProfile --sort cumulative
nvidia-smi

echo "finished "`date`" "`date +%s`""
exit 0

# Singularity containers
#SINGLRTY='/lfstev/e-938/jbonilla/sing_imgs/LuisBonillaR-singularity-master-py3_tfstable_luis.simg'
#SINGLRTY='/data/aghosh12/local-withdata.simg'
#SINGLRTY='/data/perdue/singularity/gnperdue-singularity_imgs-master-py2_tf18.simg '
#SINGLRTY='/lfstev/e-938/jbonilla/sing_imgs/LuisBonillaR-singularity-master-pyhon3_luisb.simg'
