SPM_MODEL_PATH="/data/10/litong/NICT-MT/all-4-sentencepiece-en_ja-32000.model"
BASE_PATH="/data/temp/litong/context_nmt/fairseq/"
BASE_SAVE_PATH=${1}
DATA_BASE=${BASE_PATH}/data-bin/
RESULT_FILE=${BASE_SAVE_PATH}/results.txt
mkdir -p ${BASE_SAVE_PATH}
for LANG_PAIR in $(echo "en_ja ja_en" | tr " " "\n")
do
    echo ${LANG_PAIR} >> ${RESULT_FILE}
    echo "------------------------------" >> ${RESULT_FILE}
    SAVE_PATH=${BASE_SAVE_PATH}/${LANG_PAIR}
    mkdir -p ${SAVE_PATH}
    MODEL_LIST=$(find ${BASE_PATH} -maxdepth 1 -name "*${LANG_PAIR}*bias*"| grep -v factored | sort -u | rev | cut -d"/" -f 1 | rev)
    for MODEL in $(echo ${MODEL_LIST})
    do
        echo "Model -> ${MODEL}" >> ${RESULT_FILE}
        for DATA in $(echo ${MODEL_LIST})
        do
            TARGET=${SAVE_PATH}/${MODEL}_${DATA}
            REFERENCE=${SAVE_PATH}/${MODEL}_${DATA}.ref
            MODEL_PATH=${BASE_PATH}/${MODEL}
            DATA_PATH=${DATA_BASE}/${DATA}
            fairseq-generate ${DATA_PATH} --path ${MODEL_PATH}/checkpoint_best.pt \
                --beam 6 --remove-bpe --user-dir context_nmt \
                --batch-size 512 | tee /tmp/gen.out
            grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
            grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
            spm_decode --model=${SPM_MODEL_PATH} --input_format=piece < /tmp/gen.out.sys > /tmp/gen.out.sys.retok
            spm_decode --model=${SPM_MODEL_PATH} --input_format=piece < /tmp/gen.out.ref > /tmp/gen.out.ref.retok
            if [ "$LANG_PAIR" = "en_ja" ]; then
                mecab -O wakati < /tmp/gen.out.sys.retok > ${TARGET}
                mecab -O wakati < /tmp/gen.out.ref.retok > ${REFERENCE}
                # jumanpp --segment -o ${TARGET} /tmp/gen.out.sys.retok
                # jumanpp --segment -o ${REFERENCE} /tmp/gen.out.ref.retok
            else
                cp /tmp/gen.out.sys.retok ${TARGET}
                cp /tmp/gen.out.ref.retok ${REFERENCE}
            fi
            SCORE=$(sacrebleu -b -w 2 ${REFERENCE} < ${TARGET})
            print "\tData -> ${DATA}: ${SCORE}" | tee -a ${RESULT_FILE}
        done
    done
    echo >> ${RESULT_FILE}
done
