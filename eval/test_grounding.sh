MODEL_DIR="/path_to_the_timesuite_root_folder/download/parameters"
MODEL_TYPE="VideoChat2_it4_mistral_LinearProAda"

TASK='vhd'
SPLIT='val'
DATASET='qvhighlights'
PROMPT_FILE="/path_to_the_timesuite_root_folder/prompts/vhd_description_zeroshot_new.txt"
ANNO_DIR='/path_to_the_timesuite_root_folder/dataset/TimeIT/video_highlight_detection/qvhighlights/annotations_raw'
GT_FILE="${ANNO_DIR}/highlight_${SPLIT}_release.jsonl"
VIDEO_DIR='pnorm2:s3://qvhighlight/videos'

sleep 2
MODEL_PTH="timesuite"
PTH_DIR="${MODEL_DIR}/${MODEL_PTH}.pth"
RESULT_DIR="${MODEL_DIR}/${TASK}_${SPLIT}_${MODEL_PTH}_new"

if [ ! -d "${RESULT_DIR}" ] && [ -f "${PTH_DIR}" ]; then
    mkdir -p ${RESULT_DIR}
    echo "Created directory: ${RESULT_DIR}"
    python3 /path_to_the_timesuite_root_folder/eval/eval_infer.py \
    --task=${TASK} \
    --split=${SPLIT} \
    --dataset=${DATASET} \
    --prompt_file=${PROMPT_FILE} \
    --anno_path=${ANNO_DIR} \
    --video_path=${VIDEO_DIR} \
    --model_dir=${MODEL_DIR} \
    --model_pth=${MODEL_PTH} \
    --model_type=${MODEL_TYPE} \
    --output_dir=${RESULT_DIR} \
    2>&1 | tee ${RESULT_DIR}/eval_inference.log
else
    echo "Directory ${RESULT_DIR} already exists or ${PTH_DIR} not exists. Skipping eval operation."
fi

TASK='tvg'
SPLIT='test'
DATASET='charades'
PROMPT_FILE="/path_to_the_timesuite_root_folder/prompts/tvg_description_zeroshot.txt"
ANNO_DIR='/path_to_the_timesuite_root_folder/dataset/TimeIT/temporal_video_grounding/charades/charades_annotation'
GT_FILE="${ANNO_DIR}/${SPLIT}.caption_coco_format.json"
VIDEO_DIR="pnorm2zxy:s3://zengxiangyu/Charades/"


sleep 2
MODEL_PTH="timesuite"
PTH_DIR="${MODEL_DIR}/${MODEL_PTH}.pth"
RESULT_DIR="${MODEL_DIR}/${TASK}_${SPLIT}_${MODEL_PTH}"


if [ ! -d "${RESULT_DIR}" ] && [ -f "${PTH_DIR}" ]; then
    mkdir -p ${RESULT_DIR}
    echo "Created directory: ${RESULT_DIR}"
    python3 /path_to_the_timesuite_root_folder/eval/eval_infer.py \
        --task=${TASK} \
        --split=${SPLIT} \
        --dataset=${DATASET} \
        --prompt_file=${PROMPT_FILE} \
        --anno_path=${ANNO_DIR} \
        --video_path=${VIDEO_DIR} \
        --model_dir=${MODEL_DIR} \
        --model_pth=${MODEL_PTH} \
        --model_type=${MODEL_TYPE} \
        --output_dir=${RESULT_DIR} \
        2>&1 | tee ${RESULT_DIR}/eval_inference.log
else
    echo "Directory ${RESULT_DIR} already exists or ${PTH_DIR} not exists. Skipping eval operation."
fi