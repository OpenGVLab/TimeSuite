MODEL_DIR="/path_to_the_timesuite_root_folder/download/parameters"



TASK='tvg'
SPLIT='test'
DATASET='charades'
ANNO_DIR='/path_to_the_timesuite_root_folder/dataset/TimeIT/temporal_video_grounding/charades/charades_annotation'
GT_FILE="${ANNO_DIR}/${SPLIT}.caption_coco_format.json"

sleep 1
MODEL_PTH="timesuite"
RESULT_DIR="${MODEL_DIR}/${TASK}_${SPLIT}_${MODEL_PTH}"
PRED_FILE="${RESULT_DIR}/fmt_${DATASET}_${SPLIT}_clipF8_result.json"

if [ -f "${PRED_FILE}" ]; then
    cd metrics/${TASK}
    python eval_${TASK}.py \
        --gt_file ${GT_FILE} \
        --pred_file ${PRED_FILE} \
        2>&1 | tee ${RESULT_DIR}/grounding_result.txt
    cd ../..
else
    echo "File ${PRED_FILE} not exists. Skipping eval operation."
fi


TASK='vhd'
SPLIT='val'
DATASET='qvhighlights'
ANNO_DIR='/path_to_the_timesuite_root_folder/dataset/TimeIT/video_highlight_detection/qvhighlights/annotations_raw'
GT_FILE="${ANNO_DIR}/highlight_${SPLIT}_release.jsonl"

sleep 1
MODEL_PTH="timesuite"
RESULT_DIR="${MODEL_DIR}/${TASK}_${SPLIT}_${MODEL_PTH}"
PRED_FILE="${RESULT_DIR}/fmt_${DATASET}_${SPLIT}_clipF8_result.json"

if [ -f "${PRED_FILE}" ]; then
    cd metrics/${TASK}
    python eval_${TASK}.py \
        --gt_file ${GT_FILE} \
        --pred_file ${PRED_FILE} \
        2>&1 | tee ${RESULT_DIR}/grounding_result.txt
    cd ../..
else
    echo "File ${PRED_FILE} not exists. Skipping eval operation."
fi