python gokart_runner.py --local-scheduler --module context_nmt \
    context_nmt.RunFairseqTraining \
    --train-dataset-names '["news-commentary", "europarl-v9", "rapid"]' \
    --train-source-paths '["/data/10/litong/news/news-commentary-v14.de-en.tsv", "/data/10/litong/europarl-v9.de-en.tsv", {"en": "/data/10/litong/rapid/rapid2019.de-en.en", "de": "/data/10/litong/rapid/rapid2019.de-en.de"}]' \
    --valid-dataset-names '["wmt2018"]' \
    --valid-source-paths '[{"en": "/data/10/litong/wmt2019/test/newstest2019-ende-src.en.sgm", "de": "/data/10/litong/wmt2019/test/newstest2019-ende-ref.de.sgm"}]' \
    --test-dataset-names '["wmt2019"]' \
    --test-source-paths '[{"en": "/data/10/litong/wmt2019/test/newstest2019-ende-src.en.sgm", "de": "/data/10/litong/wmt2019/test/newstest2019-ende-ref.de.sgm"}]' \
    --noisy-dataset-names '["paracrawl"]' \
    --noisy-dataset-source-paths '["/data/10/litong/paracrawl/de-en.bicleaner07.tsv"]' \
    --use-original-lr-scheduler \
    --sentencepiece-model-path "/data/10/litong/NICT-MT/sentencepiece-en_de-${1}.model" \
    --vocab-size ${1} \
    --source-lang ${2} --target-lang ${3} --data-mode ${4} \
    --experiment-path ${5} \
    --preprocess-workers 8 --batch-size ${6} --save-interval 2000 \
    --context-bias ${7} \
    --source-max-sequence-length ${8} --target-max-sequence-length ${9} \
    ${10}
