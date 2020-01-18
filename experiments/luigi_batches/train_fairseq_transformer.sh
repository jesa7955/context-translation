python gokart_runner.py --local-scheduler --module context_nmt \
    context_nmt.${1} \
    --train-dataset-names '["jiji", "onto", "ami", "conver_train"]' \
    --train-source-paths '["/data/11/nict_dl_data/jiji-parallel-corpus-v1", "/data/11/nict_dl_data/json_ontonotes", "/data/11/nict_dl_data/json_ami_corpus", "/data/11/nict_dl_data/json_business_conversation_split/train.json"]' \
    --valid-dataset-names '["conver_valid"]' \
    --valid-source-paths '["/data/11/nict_dl_data/json_business_conversation_split/dev.json"]' \
    --test-dataset-names '["conver_test"]' \
    --test-source-paths '["/data/11/nict_dl_data/json_business_conversation_split/test.json"]' \
    --sentencepiece-model-path "/data/10/litong/NICT-MT/all-4-sentencepiece-en_ja-${2}.model" \
    --vocab-size ${2} \
    --source-lang ${3} --target-lang ${4} --data-mode ${5} \
    --experiment-path ${6} --cuda-device ${7} \
    --preprocess-workers 8 --batch-size 2048 --save-interval 4000
