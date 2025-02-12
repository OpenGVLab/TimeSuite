export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"
ROOT_DIR=/path_to_the_timesuite_root_folder
PARTITION='video5'
TRAIN_TYPE=""


MODEL_TYPE='LinearP'
JOB_NAME="F192_CF8${TRAIN_TYPE}_${MODEL_TYPE}_TimePro_Normal"

# MODEL_TYPE='LinearProAda'
# JOB_NAME="F128_CF8${TRAIN_TYPE}_${MODEL_TYPE}_TimePro_Normal"

NNODE=2
NUM_GPUS=8
NUM_CPUS=128


OUTPUT_DIR="${ROOT_DIR}/$(dirname $0)/${JOB_NAME}"
echo "Model Dir : ${OUTPUT_DIR}"
mkdir ${OUTPUT_DIR}




# srun -p ${PARTITION} \
#     --job-name=${JOB_NAME} \
#     -n${NNODE} \
#     --gres=gpu:${NUM_GPUS} \
#     --ntasks-per-node=1 \
#     --cpus-per-task=${NUM_CPUS} \

bash torchrun.sh \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    tasks/train_it4${TRAIN_TYPE}.py \
    $(dirname $0)/config_${MODEL_TYPE}${TRAIN_TYPE}.py \
    output_dir ${OUTPUT_DIR} \
    2>&1 | tee ${OUTPUT_DIR}/bash_output.log