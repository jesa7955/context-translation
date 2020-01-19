LANG_PAIR="en_ja"
MODEL_BASE="jiji_context_filter"
CACHE="/home/litong/context_translation/resources/"
SOURCE="/home/litong/context_translation/experiments/allennlp_configs/context_filter/jiji/${LANG_PAIR}/${MODEL_BASE}.jsonnet"
VAL_SOURCE="/home/litong/context_translation/experiments/allennlp_configs/context_filter/jiji/${LANG_PAIR}/${MODEL_BASE}_val_w5.jsonnet"
TARGET_BASE="/data/local/litong/context_nmt/"
CONTEXT_INDICATOR="null"
TRAIN_FILE="123"
VALID_FILE="456"
TEST_FILE="789"
VALID_CONFIG=${CACHE}/valid.json
for ITER in $(seq 1 3);
do
    TRAIN_DIR=${TARGET_BASE}/${MODEL_BASE}_${ITER}
    TRAIN_CONFIG=${CACHE}/${MODEL_BASE}_${LANG_PAIR}_train_${ITER}.json
    INDEXER_TARGET=${TARGET_BASE}/context_sentence_indexes/${MODEL_BASE}_${ITER}
    CONTEXT_INDICATOR=$(echo ${CONTEXT_INDICATOR} | sed 's/\//\\\//g')
    jsonnet ${SOURCE} | sed "s/\"context_sentence_index_file\": null/\"context_sentence_index_file\": \"${CONTEXT_INDICATOR}\"/g" | sed "s/\"null\"/null/g" > ${TRAIN_CONFIG}
    allennlp train -s ${TRAIN_DIR} --include-package context_nmt ${TRAIN_CONFIG}
    jsonnet ${VAL_SOURCE} > ${TRAIN_DIR}/config.json
    for FILE in $(echo "$TRAIN_FILE ${VALID_FILE} ${TEST_FILE}" | tr " " "\n")
    do
        allennlp predict --include-package context_nmt --output-file ${TEMP} --batch-size 1024 --cuda-device 0 \
                         --use-dataset-reader --predictor context_sentence_filter --silent ${TRAIN_DIR} ${FILE}
        cat ${TEMP} >> ${INDEXER_TARGET}
    done
    CONTEXT_INDICATOR=${INDEXER_TARGET}
done
