export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

PARTITION='video5'
JOB_NAME='stage4_ds_TimeIT'
NNODE=1
NUM_GPUS=8
NUM_CPUS=96
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"

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
    tasks/train_it4_ds.py \
    $(dirname $0)/config_7b_stage4_ds.py \
    output_dir ${OUTPUT_DIR} \
    2>&1 | tee ${OUTPUT_DIR}/${JOB_NAME}_bash_output.log
