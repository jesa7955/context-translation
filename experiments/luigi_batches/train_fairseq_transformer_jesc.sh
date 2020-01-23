CUDA_VISIBLE_DEVICES=${1} python gokart_runner.py --local-scheduler --module context_nmt \
    context_nmt.RunFairseqTraining \
    --train-dataset-names '["jesc"]' \
    --train-source-paths '["/data/10/litong/jesc/train"]' \
    --valid-dataset-names '["jesc"]' \
    --valid-source-paths '["/data/10/litong/jesc/dev"]' \
    --test-dataset-names '["jesc"]' \
    --test-source-paths '["/data/10/litong/jesc/test"]' \
    --sentencepiece-model-path "/data/10/litong/NICT-MT/jesc-sentencepiece-en_ja-32000.model" \
    --vocab-size 32000 --data-format "tsv" \
    --source-lang ${2} --target-lang ${3} --data-mode "1-to-1" \
    --experiment-path /data/temp/litong/context_nmt/fairseq \
    --preprocess-workers 8 --batch-size 4096 --save-interval 2000 \
    --sentence-level
