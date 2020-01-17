CUDA_VISIBLE_DEVICES=1 fairseq-generate ${1} --path ${2} --beam 6 --batch-size 1024 --remove-bpe | tee /tmp/gen.out
grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
sed -i 's/\ //g' /tmp/gen.out.sys
sed -i 's/\ //g' /tmp/gen.out.ref
sed -i 's/▁//g' /tmp/gen.out.sys
sed -i 's/▁//g' /tmp/gen.out.ref
# jumanpp --segment -o /tmp/gen.out.tok.sys /tmp/gen.out.sys
# jumanpp --segment -o /tmp/gen.out.tok.ref /tmp/gen.out.ref
mecab -O wakati < /tmp/gen.out.sys > /tmp/gen.out.tok.sys
mecab -O wakati < /tmp/gen.out.ref > /tmp/gen.out.tok.ref
sacrebleu -b -w 2 /tmp/gen.out.tok.ref < /tmp/gen.out.tok.sys
