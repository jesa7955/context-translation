TARGET_FILE="/home/litong/context_translation/resources/context_filter/jiji_context_filter_ja_en_0"
TARGET_NAME="jiji_context_filter_ja_en_0"
# Wait for the training of logits based model
while [ $(ps ax | grep ${TARGET_NAME} | grep -c "fairseq-train") -lt 1 ]; do
    echo "waiting for the logits based model to run"
    sleep 60
done
# Wait for a GPU to use
while [ $(nvidia-smi | grep python | wc -l) -eq 4 ]; do
    echo "waiting for gpu"
    sleep 60
done
# Find us a GPU
GPUS=$(nvidia-smi | grep python | cut -f 5 -d" ")
for GPU in $(seq 0 3); do
    if [ $(echo ${GPUS} | grep -c ${GPU}) -eq 0 ]; then
        break
    fi
done
CUDA_VISIBLE_DEVICES=${GPU} bash experiments/luigi_batches/train_fairseq_transformer.sh 32000 ja en 2-to-1 \
    /data/temp/litong/context_nmt/fairseq 1 "--context-sentence-index-file ${TARGET_FILE}" "--context-metric-key probs"

# jsonnet /home/litong/context_translation/experiments/allennlp_configs/context_filter/jiji/ja_en/jiji_context_filter_val_ws5.jsonnet > /data/local/litong/context_nmt/jiji_context_filter_ja_en_0/config.json
# allennlp predict --include-package context_nmt --output-file /data/local/litong/context_nmt/context_sentence_indexes/jiji_context_filter_ja_en_0 \
#     --batch-size 900 --cuda-device 1 --use-dataset-reader --predictor context_sentence_filter \
#     --silent /data/local/litong/context_nmt/jiji_context_filter_ja_en_0 \
#     /home/litong/context_translation/resources/jiji_onto_ami_conver_train_conver_valid_conver_test_ee753355f7a5dab8b4c5ed2281b37f76.pkl
