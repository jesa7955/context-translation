READY_FILE="resources/jiji_context_filter_cleaned_limited_ja_en_0_ready"
TARGET_FILE="/home/litong/context_translation/resources/context_filter/jiji_context_filter_cleaned_limited_ja_en_0"
while [ $(nvidia-smi | grep 23870 | wc -l) -eq 1 ]; do
    echo "waiting for the source file to ready"
    sleep 60
done
cp /data/local/litong/context_nmt/context_sentence_indexes/jiji_context_filter_cleaned_limited_ja_en_0 ${TARGET_FILE}

touch ${READY_FILE}

# jsonnet /home/litong/context_translation/experiments/allennlp_configs/context_filter/jiji/ja_en/jiji_context_filter_val_ws5.jsonnet > /data/local/litong/context_nmt/jiji_context_filter_ja_en_0/config.json
# allennlp predict --include-package context_nmt --output-file /data/local/litong/context_nmt/context_sentence_indexes/jiji_context_filter_ja_en_0 \
#     --batch-size 900 --cuda-device 1 --use-dataset-reader --predictor context_sentence_filter \
#     --silent /data/local/litong/context_nmt/jiji_context_filter_ja_en_0 \
#     /home/litong/context_translation/resources/jiji_onto_ami_conver_train_conver_valid_conver_test_ee753355f7a5dab8b4c5ed2281b37f76.pkl
