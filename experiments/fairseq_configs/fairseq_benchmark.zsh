SPM_MODEL_PATH="/data/10/litong/NICT-MT/all-4-sentencepiece-en_ja-32000.model"
BASE_PATH="/data/temp/litong/context_nmt/fairseq/"
CUDA_VISIBLE_DEVICES=${1}
BASE_SAVE_PATH=${2}
DATA_BASE=${BASE_PATH}/data-bin/
RESULT_FILE=${BASE_SAVE_PATH}/results.txt
mkdir -p ${BASE_SAVE_PATH}
for LANG_PAIR in $(echo "en_ja ja_en" | tr " " "\n")
do
    echo ${LANG_PAIR} >> ${RESULT_FILE}
    echo "------------------------------" >> ${RESULT_FILE}
    SOURCE_LANG=$(echo ${LANG_PAIR} | cut -d"_" -f 1)
    TARGET_LANG=$(echo ${LANG_PAIR} | cut -d"_" -f 2)
    SAVE_PATH=${BASE_SAVE_PATH}/${LANG_PAIR}
    mkdir -p ${SAVE_PATH}
    MODEL_LIST=$(find ${BASE_PATH} -maxdepth 1 -name "*${LANG_PAIR}*bias*"| grep -v factored | sort -u | rev | cut -d"/" -f 1 | rev)
    for MODEL in $(echo ${MODEL_LIST})
    do
        echo "Model -> ${MODEL}" >> ${RESULT_FILE}
        for DATA in $(ls -lt resources/test*${SOURCE_LANG} | head -12 | sort -k 5 --reverse | head -5 | cut -d" " -f 9)
        do
            DATA_NAME="Bias_$(head -5 ${DATA} | grep -c "^@@CONCAT@@")"
            REFERENCE_DATA=$(echo ${DATA} | cut -d"." -f 1).${TARGET_LANG}
            TARGET=${SAVE_PATH}/${MODEL}_${DATA_NAME}
            REFERENCE=${TARGET}.ref
            MODEL_PATH=${BASE_PATH}/${MODEL}
            DATA_PATH=${DATA_BASE}/${MODEL}
            cat ${DATA} | fairseq-interactive --path ${MODEL_PATH}/checkpoint_best.pt \
                --beam 6 --remove-bpe --user-dir context_nmt --buffer-size 512 --batch-size 256 \
                --source-lang ${SOURCE_LANG} --target-lang ${TARGET_LANG} ${DATA_PATH} | tee /tmp/gen.out
            grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
            spm_decode --model=${SPM_MODEL_PATH} --input_format=piece < /tmp/gen.out.sys > /tmp/gen.out.sys.retok
            spm_decode --model=${SPM_MODEL_PATH} --input_format=piece < ${REFERENCE_DATA} > /tmp/gen.out.ref.retok
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
            print "\tData -> ${DATA_NAME}: ${SCORE}" | tee -a ${RESULT_FILE}
        done
    done
    echo >> ${RESULT_FILE}
done
