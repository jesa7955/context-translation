LANG_PAIR=${1}
MODEL_BASE=${2}
CONTEXT_INDICATOR=${3}
START_ITER=${4}
END_ITER=${5}
CUDA_DEVICE=${6}
CACHE="/home/litong/context_translation/resources/"
SOURCE="/home/litong/context_translation/experiments/allennlp_configs/context_filter/jiji/${LANG_PAIR}/${MODEL_BASE}.jsonnet"
VAL_SOURCE="/home/litong/context_translation/experiments/allennlp_configs/context_filter/jiji/${LANG_PAIR}/${MODEL_BASE}_val_w5.jsonnet"
TARGET_BASE="/data/local/litong/context_nmt/"
TEMP=${TARGET_BASE}/${LANG_PAIR}_${MODEL_BASE}_TEMP
for ITER in $(seq ${START_ITER} ${END_ITER});
do
    PREFIX=${MODEL_BASE}_${LANG_PAIR}_${ITER}
    TRAIN_DIR=${TARGET_BASE}/${PREFIX}
    TRAIN_CONFIG=${CACHE}/${PREFIX}.json
    INDEXER_TARGET=${TARGET_BASE}/context_sentence_indexes/${PREFIX}
    CONTEXT_INDICATOR=$(echo ${CONTEXT_INDICATOR} | sed 's/\//\\\//g')
    jsonnet ${SOURCE} | sed "s/\"context_sentence_index_file\": null/\"context_sentence_index_file\": \"${CONTEXT_INDICATOR}\"/g" | sed "s/\"null\"/null/g" > ${TRAIN_CONFIG}
    allennlp train -s ${TRAIN_DIR} --include-package context_nmt ${TRAIN_CONFIG}
    jsonnet ${VAL_SOURCE} > ${TRAIN_DIR}/config.json
    for FILE in $(grep "pkl" ${TRAIN_CONFIG} | cut -d'"' -f 4)
    do
        allennlp predict --include-package context_nmt --output-file ${TEMP} --batch-size 1024 --cuda-device ${CUDA_DEVICE} \
                         --use-dataset-reader --predictor context_sentence_filter --silent ${TRAIN_DIR} ${FILE}
        cat ${TEMP} >> ${INDEXER_TARGET}
    done
    rm ${TEMP}
    CONTEXT_INDICATOR=${INDEXER_TARGET}
done
