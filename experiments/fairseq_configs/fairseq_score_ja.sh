SPM_MODEL_PATH="/data/10/litong/NICT-MT/all-4-sentencepiece-en_ja-32000.model"
#CUDA_VISIBLE_DEVICES=${1}
fairseq-generate ${1} --path ${2} --beam 6 --remove-bpe --user-dir context_nmt --batch-size ${3} --gen-subset ${4} | tee /tmp/gen.out
grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
spm_decode --model=${SPM_MODEL_PATH} --input_format=piece < /tmp/gen.out.sys > /tmp/gen.out.sys.retok
spm_decode --model=${SPM_MODEL_PATH} --input_format=piece < /tmp/gen.out.ref > /tmp/gen.out.ref.retok
# jumanpp --segment -o /tmp/gen.out.tok.sys /tmp/gen.out.sys
# jumanpp --segment -o /tmp/gen.out.tok.ref /tmp/gen.out.ref
mecab -O wakati < /tmp/gen.out.sys.retok > /tmp/gen.out.sys.tok
mecab -O wakati < /tmp/gen.out.ref.retok > /tmp/gen.out.ref.tok
sacrebleu -b -w 2 /tmp/gen.out.ref.tok < /tmp/gen.out.sys.tok
