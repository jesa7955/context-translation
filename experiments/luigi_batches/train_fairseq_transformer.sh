python gokart_runner.py --local-scheduler --module context_nmt \
    context_nmt.RunFairseqTraining \
    --train-dataset-names '["jiji", "onto", "ami", "conver_train"]' \
    --train-source-paths '["/data/11/nict_dl_data/jiji-parallel-corpus-v1", "/data/11/nict_dl_data/json_ontonotes", "/data/11/nict_dl_data/json_ami_corpus", "/data/11/nict_dl_data/json_business_conversation_split/train.json"]' \
    --valid-dataset-names '["conver_valid"]' \
    --valid-source-paths '["/data/11/nict_dl_data/json_business_conversation_split/dev.json"]' \
    --test-dataset-names '["conver_test"]' \
    --test-source-paths '["/data/11/nict_dl_data/json_business_conversation_split/test.json"]' \
    --sentencepiece-model-path "/data/10/litong/NICT-MT/all-4-sentencepiece-en_ja-${1}.model" \
    --vocab-size ${1} \
    --source-lang ${2} --target-lang ${3} --data-mode ${4} \
    --experiment-path ${5} \
    --preprocess-workers 8 --batch-size 4096 --save-interval 2000 \
    --context-bias ${6} \
    ${7} ${8}
