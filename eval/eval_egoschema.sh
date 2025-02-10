ROOT_DIR="/path_to_the_timesuite_root_folder/download/parameters"

python3 eval/validate_egoschema.py \
    --f ${ROOT_DIR}/Egoschema_test_timesuite/result.json \
    2>&1 | tee "${ROOT_DIR}/Egoschema_test_timesuite/acc.txt"
