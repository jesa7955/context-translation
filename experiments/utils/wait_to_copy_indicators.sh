PID=3273
READY_FILE="resources/${PID}_ready"
TARGET_FILE="/data/local/litong/context_nmt/context_sentence_indexes/jiji_context_filter_translated_full_smaller_lr_en_ja_0"
while [ $(nvidia-smi | grep ${PID} | wc -l) -eq 1 ]; do
    echo "waiting for the source file to ready"
    sleep 60
done
cp ${TARGET_FILE} resources/context_filter/

touch ${READY_FILE}

# jsonnet /home/litong/context_translation/experiments/allennlp_configs/context_filter/jiji/ja_en/jiji_context_filter_val_ws5.jsonnet > /data/local/litong/context_nmt/jiji_context_filter_ja_en_0/config.json
# allennlp predict --include-package context_nmt --output-file /data/local/litong/context_nmt/context_sentence_indexes/jiji_context_filter_ja_en_0 \
#     --batch-size 900 --cuda-device 1 --use-dataset-reader --predictor context_sentence_filter \
#     --silent /data/local/litong/context_nmt/jiji_context_filter_ja_en_0 \
#     /home/litong/context_translation/resources/jiji_onto_ami_conver_train_conver_valid_conver_test_ee753355f7a5dab8b4c5ed2281b37f76.pkl
