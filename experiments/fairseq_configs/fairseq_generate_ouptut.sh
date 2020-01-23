SPM_MODEL_PATH="/data/10/litong/NICT-MT/all-4-sentencepiece-en_ja-32000.model"
BASE_PATH="/data/temp/litong/context_nmt/fairseq/"
CUDA_DEVICE=${1}
BATCH_SIZE=${2}
BASE_SAVE_PATH=${3}
DATA_BASE=${BASE_PATH}/data-bin/
RESULT_FILE=${BASE_SAVE_PATH}/results.txt
mkdir -p ${BASE_SAVE_PATH}
for LANG_PAIR in $(echo "en_ja ja_en" | tr " " "\n")
do
    echo ${LANG_PAIR} >> ${RESULT_FILE}
    echo "------------------------------" >> ${RESULT_FILE}
    SAVE_PATH=${BASE_SAVE_PATH}/${LANG_PAIR}
    mkdir -p ${SAVE_PATH}
    for MODEL in $(find ${BASE_PATH} -maxdepth 1 | grep "${LANG_PAIR}" | sort -u | rev | cut -d"/" -f 1 | rev | grep -v factored)
    do
        TARGET=${SAVE_PATH}/${MODEL}
        REFERENCE=${SAVE_PATH}/${MODEL}.ref
        CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} fairseq-generate ${DATA_BASE}/${MODEL} \
            --path ${BASE_PATH}/${MODEL}/checkpoint_best.pt --beam 6 --remove-bpe \
            --user-dir context_nmt --batch-size ${BATCH_SIZE} | tee /tmp/gen.out
        grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
        grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
        spm_decode --model=${SPM_MODEL_PATH} --input_format=piece < /tmp/gen.out.sys > /tmp/gen.out.sys.retok
        spm_decode --model=${SPM_MODEL_PATH} --input_format=piece < /tmp/gen.out.ref > /tmp/gen.out.ref.retok
        if [ "$LANG_PAIR" == "en_ja" ]; then
            mecab -O wakati < /tmp/gen.out.sys.retok > ${TARGET}
            mecab -O wakati < /tmp/gen.out.ref.retok > ${REFERENCE}
            # jumanpp --segment -o ${TARGET} /tmp/gen.out.sys.retok
            # jumanpp --segment -o ${REFERENCE} /tmp/gen.out.ref.retok
        else
            cp /tmp/gen.out.sys.retok ${TARGET}
            cp /tmp/gen.out.ref.retok ${REFERENCE}
        fi
        echo "${MODEL}: $(sacrebleu -b -w 2 ${REFERENCE} < ${TARGET})" >> ${RESULT_FILE}
    done
    echo >> ${RESULT_FILE}
done
