SPM_MODEL_PATH="/data/10/litong/NICT-MT/all-4-sentencepiece-en_ja-32000.model"
BASE_PATH="/data/temp/litong/fairseq/data-bin"
#CUDA_VISIBLE_DEVICES=${1}
BASE_SAVE_PATH=${1}
RESULT_FILE=${BASE_SAVE_PATH}/results.txt
mkdir -p ${BASE_SAVE_PATH}
for LANG_PAIR in $(echo "en_ja ja_en" | tr " " "\n")
do
    echo ${LANG_PAIR} >> ${RESULT_FILE}
    echo "------------------------------" >> ${RESULT_FILE}
    SAVE_PATH=${BASE_SAVE_PATH}/${LANG_PAIR}
    mkdir -p ${SAVE_PATH}
    # for MODEL in $(find ${BASE_PATH} -maxdepth 1 | grep "${LANG_PAIR}" | sort -u | grep filtered)
    for MODEL in $(find ${BASE_PATH} -maxdepth 1 | grep "${LANG_PAIR}" | sort -u)
    do
        MODEL_NAME=$(echo ${MODEL} | rev | cut -d"/" -f 1 | rev)
        echo ${MODEL_NAME} >> ${RESULT_FILE}
        SCORE=""
        for SPLIT in $(echo "valid test" | tr " " "\n")
        do
            TARGET=${SAVE_PATH}/${MODEL_NAME}_${SPLIT}
            REFERENCE=${TARGET}.ref
            fairseq-generate  ${MODEL} --path ${MODEL}/../../${MODEL_NAME}/checkpoint_best.pt --beam 6 \
                --user-dir context_nmt --batch-size 500 --gen-subset ${SPLIT} | tee /tmp/gen.out
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
            SCORE="${SCORE}@${SPLIT}: $(sacrebleu -b -w 2 ${REFERENCE} < ${TARGET})"
        done
        echo ${SCORE} | tr "@" "\t" >> ${RESULT_FILE}
    done
    echo >> ${RESULT_FILE}
done
