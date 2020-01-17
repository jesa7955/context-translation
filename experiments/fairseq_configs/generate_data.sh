python gokart_runner.py --local-scheduler --module context_nmt \
    context_nmt.GenerateConversationSplits \
    --train-dataset-names '["onto", "ami", "conver_train"]' \
    --train-source-paths '["/data/11/nict_dl_data/json_ontonotes", "/data/11/nict_dl_data/json_ami_corpus", "/data/11/nict_dl_data/json_business_conversation_split_bkp/train.json"]' \
    --valid-dataset-names '["conver_valid"]' \
    --valid-source-paths '["/data/11/nict_dl_data/json_business_conversation_split_bkp/dev.json"]' \
    --test-dataset-names '["conver_test"]' \
    --test-source-paths '["/data/11/nict_dl_data/json_business_conversation_split_bkp/test.json"]' \
    --sentencepiece-model-path '/data/10/litong/NICT-MT/all-3-sentencepiece-en_ja.model' \
    --shared-vocab \
