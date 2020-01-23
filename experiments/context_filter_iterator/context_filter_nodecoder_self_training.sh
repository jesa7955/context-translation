LANG_PAIR=${1}
CONTEXT_INDICATOR=${2}
START_ITER=${3}
END_ITER=${4}
CUDA_DEVICE=${5}
TARGET_FILE=${6}
BATCH_SIZE=${7}
CACHE="/home/litong/context_translation/resources/"
SOURCE="/home/litong/context_translation/experiments/allennlp_configs/context_filter/jiji/${LANG_PAIR}/jiji_context_filter_nodecoder.jsonnet"
VAL_SOURCE="/home/litong/context_translation/experiments/allennlp_configs/context_filter/jiji/${LANG_PAIR}/jiji_context_filter_nodecoder_val_ws5.jsonnet"
TARGET_BASE="/data/local/litong/context_nmt/"
for ITER in $(seq ${START_ITER} ${END_ITER});
do
    PREFIX=jiji_context_filter_${LANG_PAIR}_${ITER}
    TRAIN_DIR=${TARGET_BASE}/${PREFIX}
    TRAIN_CONFIG=${CACHE}/${PREFIX}.json
    INDEXER_TARGET=${TARGET_BASE}/context_sentence_indexes/${PREFIX}
    CONTEXT_INDICATOR=$(echo ${CONTEXT_INDICATOR} | sed 's/\//\\\//g')
    jsonnet ${SOURCE} | sed "s/\"context_sentence_index_file\": null/\"context_sentence_index_file\": \"${CONTEXT_INDICATOR}\"/g" | sed "s/\"null\"/null/g" > ${TRAIN_CONFIG}
    allennlp train -s ${TRAIN_DIR} --include-package context_nmt ${TRAIN_CONFIG} -f
    jsonnet ${VAL_SOURCE} > ${TRAIN_DIR}/config.json
    allennlp predict --include-package context_nmt --output-file ${INDEXER_TARGET} --batch-size ${BATCH_SIZE} --cuda-device ${CUDA_DEVICE} \
                     --use-dataset-reader --predictor context_sentence_filter --silent ${TRAIN_DIR} ${TARGET_FILE}
    CONTEXT_INDICATOR=${INDEXER_TARGET}
done
