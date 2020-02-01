SPM_MODEL_PATH="/data/10/litong/NICT-MT/all-4-sentencepiece-en_ja-32000.model"
BASE_PATH="/data/temp/litong/context_nmt/fairseq_temp/"
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
    # for MODEL in $(find ${BASE_PATH} -maxdepth 1 -name "*${LANG_PAIR}*"| grep -v factored | grep -v "1-to-1" | sort -u | rev | cut -d"/" -f 1 | rev)
    for MODEL in $(find ${BASE_PATH} -maxdepth 1 -name "*${LANG_PAIR}*"| grep -v factored | sort -u | rev | cut -d"/" -f 1 | rev)
    do
        echo "Model -> ${MODEL}" >> ${RESULT_FILE}
        for DATA in $(ls -tl resources/test*${SOURCE_LANG} | grep "Jan 24\|Jan 25" | tail -10 | sort -k 5 --reverse | cut -d" " -f 9 | head -5)
        do
            DATA_NAME="Previous_$(head -5 ${DATA} | grep -c "^@@CONCAT@@")"
            SCORE=""
            for SPLIT in $(echo "valid\ntest")
            do
                REFERENCE_DATA=$(echo ${DATA} | sed "s/test/${SPLIT}/g"| cut -d"." -f 1).${TARGET_LANG}
                TARGET=${SAVE_PATH}/${MODEL}_${DATA_NAME}_${SPLIT}
                REFERENCE=${TARGET}.ref
                MODEL_PATH=${BASE_PATH}/${MODEL}
                DATA_PATH=${DATA_BASE}/${MODEL}
                cat $(echo ${DATA} | sed "s/test/${SPLIT}/g") | fairseq-interactive --path ${MODEL_PATH}/checkpoint_best.pt \
                    --beam 6 --user-dir context_nmt --buffer-size 512 --batch-size 256 \
                    --source-lang ${SOURCE_LANG} --target-lang ${TARGET_LANG} ${DATA_PATH} | tee /tmp/gen.out
                grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
                spm_decode --model=${SPM_MODEL_PATH} --input_format=piece < /tmp/gen.out.sys > /tmp/gen.out.sys.retok
                spm_decode --model=${SPM_MODEL_PATH} --input_format=piece < ${REFERENCE_DATA} > /tmp/gen.out.ref.retok
                if [ "$LANG_PAIR" = "en_ja" ]; then
                    mecab -O wakati < /tmp/gen.out.sys.retok > ${TARGET}
                    mecab -O wakati < /tmp/gen.out.ref.retok > ${REFERENCE}
                else
                    cp /tmp/gen.out.sys.retok ${TARGET}
                    cp /tmp/gen.out.ref.retok ${REFERENCE}
                fi
                SCORE="${SCORE}\t${SPLIT}: $(sacrebleu -b -w 2 ${REFERENCE} < ${TARGET})"
            done
            echo "\tDATA -> ${DATA_NAME}: ${SCORE}" | tee -a ${RESULT_FILE}
        done
    done
    echo >> ${RESULT_FILE}
done
