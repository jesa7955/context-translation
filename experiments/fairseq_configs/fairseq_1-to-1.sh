CODE_ROOT="/home/litong/context_translation"
fairseq-preprocess --source-lang ${3} --target-lang ${4} \
    --trainpref ${CODE_ROOT}/resources/train_${1} \
    --validpref ${CODE_ROOT}/resources/valid_${1} \
    --testpref ${CODE_ROOT}/resources/test_${1} \
    --destdir /data/local/litong/context_nmt/fairseq/data-bin/${2}.${3}-${4} \
    --joined-dictionary --workers 7
CUDA_VISIBLE_DEVICES=0 fairseq-train  \
    /data/local/litong/context_nmt/fairseq/data-bin/${2}.${3}-${4}/ \
    --tokenizer space \
    --arch transformer --share-decoder-input-output-embed --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)'  \
    --lr 1e-4 --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.7 \
    --dropout 0.2 --attention-dropout 0.2 --activation-dropout 0.2 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --encoder-normalize-before --decoder-normalize-before \
    --max-tokens 8192 \
    --save-interval-updates 1000 \
    --no-last-checkpoints \
    --save-dir /data/local/litong/context_nmt/fairseq/1-to-1-transformer-${2} \
    --tensorboard-logdir /data/local/litong/context_nmt/fairseq/1-to-1-transformer-${2}/tensorboard_log \
    --patience 10 --fp16 --reset-optimizer
