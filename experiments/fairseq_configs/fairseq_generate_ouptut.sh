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
    for MODEL in $(find ${BASE_PATH} -maxdepth 1 | grep "${LANG_PAIR}" | sort -u | rev | cut -d"/" -f 1 | rev | grep -v factored | grep -v filtered)
    do
        echo ${MODEL} >> ${RESULT_FILE}
        SCORE=""
        for SPLIT in $(echo "valid test" | tr " " "\n")
        do
            MODEL_BIAS=$(echo ${MODEL} | rev | cut -f 1 -d"_")
            BIAS_LIST=$(seq 1 5)
            DATA_LIST=$(ls -tl resources/${SPLIT}*${SOURCE_LANG} | grep "Jan 24\|Jan 25" | tail -10 | sort -k 5 --reverse | cut -d" " -f 9 | head -6)
            if [ $(echo ${BIAS_LIST} | tr " " "\n" | grep -c ${MODEL_BIAS}) -eq 0 ]; then
                DATA=$(echo ${DATA_LIST} | tr " " "\n" | tail -1)
                REFERENCE_DATA=$(echo ${DATA_LIST} | tr " " "\n" | head -1 | cut -d"." -f 1).${TARGET_LANG}
            else
                for DATA in $(echo ${DATA_LIST} | tr " " "\n" | head -5)
                do
                    if [ "$(head -5 ${DATA} | grep -c "^@@CONCAT@@")" == "${MODEL_BIAS}" ]; then
                        break
                    fi
                done
                REFERENCE_DATA=$(echo ${DATA} | cut -d"." -f 1).${TARGET_LANG}
            fi
            echo ${DATA}
            TARGET=${SAVE_PATH}/${MODEL}_${SPLIT}
            REFERENCE=${TARGET}.ref
            cat ${DATA} | fairseq-interactive ${DATA_BASE}/${MODEL} \
                --path ${BASE_PATH}/${MODEL}/checkpoint_best.pt --beam 6 \
                --user-dir context_nmt --batch-size 256 --buffer-size 512 \
                --source-lang ${SOURCE_LANG} --target-lang ${TARGET_LANG} | tee /tmp/gen.out
            grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
            spm_decode --model=${SPM_MODEL_PATH} --input_format=piece < /tmp/gen.out.sys > /tmp/gen.out.sys.retok
            spm_decode --model=${SPM_MODEL_PATH} --input_format=piece < ${REFERENCE_DATA} > /tmp/gen.out.ref.retok
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
