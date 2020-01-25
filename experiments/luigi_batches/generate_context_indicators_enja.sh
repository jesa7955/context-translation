python gokart_runner.py --local-scheduler --module context_nmt context_nmt.GenerateContextIndicator \
    --sentence-translation-model-name jparacrawl_base --split-name train \
    --dataset-names '["jiji", "conver_train"]' \
    --source-paths '["/data/11/nict_dl_data/jiji-parallel-corpus-v1", "/data/11/nict_dl_data/json_business_conversation_split/train.json"]' \
    --sentence-translation-models '{"en": "/data/10/litong/jparacrawl/en-ja-base/base.pretrain.pt", "ja": "/data/10/litong/jparacrawl/ja-en-base/base.pretrain.pt"}' \
    --sentence-sentencepiece-models '{"en": "/data/10/litong/jparacrawl/enja_spm_models/spm.en.nopretok.model", "ja":"/data/10/litong/jparacrawl/enja_spm_models/spm.ja.nopretok.model"}' \
    --source-lang en --target-lang ja --context-aware-sentencepiece-model "/data/10/litong/NICT-MT/all-4-sentencepiece-en_ja-32000.model" \
    --context-aware-translation-models '{"-1": "/data/temp/litong/context_nmt/fairseq/data-bin/jiji_onto_ami_conver_train_1-to-1_en_ja/checkpoint_best.pt", "1": "/data/temp/litong/context_nmt/fairseq/data-bin/jiji_onto_ami_conver_train_2-to-1_en_ja_context_bias_1/checkpoint_best.pt", "2": "/data/temp/litong/context_nmt/fairseq/data-bin/jiji_onto_ami_conver_train_2-to-1_en_ja_context_bias_2/checkpoint_best.pt", "3": "/data/temp/litong/context_nmt/fairseq/data-bin/jiji_onto_ami_conver_train_2-to-1_en_ja_context_bias_3/checkpoint_best.pt", "4": "/data/temp/litong/context_nmt/fairseq/data-bin/jiji_onto_ami_conver_train_2-to-1_en_ja_context_bias_4/checkpoint_best.pt", "5": "/data/temp/litong/context_nmt/fairseq/data-bin/jiji_onto_ami_conver_train_2-to-1_en_ja_context_bias_5/checkpoint_best.pt"}'
    # --dataset-names '["jiji", "onto", "ami", "conver_train"]' \
    # --source-paths '["/data/11/nict_dl_data/jiji-parallel-corpus-v1", "/data/11/nict_dl_data/json_ontonotes", "/data/11/nict_dl_data/json_ami_corpus", "/data/11/nict_dl_data/json_business_conversation_split/train.json"]' \
