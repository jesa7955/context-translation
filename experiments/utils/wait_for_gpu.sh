READY_FILE="resources/jiji_context_filter_nodecoder_ja_en_0_ready"
TARGET_FILE="/home/litong/context_translation/resources/context_filter/jiji_context_filter_ja_en_0"
while [ ! -e ${READY_FILE} ]; do
    echo "waiting for indicators"
    sleep 60
done
while [ $(nvidia-smi | grep python | wc -l) -eq 4 ]; do
    echo "waiting for gpu"
    sleep 60
done
GPUS=$(nvidia-smi | grep python | cut -f 5 -d" ")
for GPU in $(seq 0 3); do
    if [ $(echo ${GPUS} | grep -c ${GPU}) -eq 0 ]; then
        break
    fi
done
CUDA_VISIBLE_DEVICES=${GPU} bash experiments/luigi_batches/train_fairseq_transformer.sh 32000 ja en 2-to-1 \
    /data/temp/litong/context_nmt/fairseq 1 "--context-sentence-index-file ${TARGET_FILE}"
