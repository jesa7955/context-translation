#CUDA_VISIBLE_DEVICES=1 fairseq-generate ${1} --path ${2} --beam 6 --batch-size 1024 --remove-bpe | tee /tmp/gen.out
grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
sed -i 's/\ //g' /tmp/gen.out.sys
sed -i 's/\ //g' /tmp/gen.out.ref
sed -i 's/▁/ /g' /tmp/gen.out.sys
sed -i 's/▁/ /g' /tmp/gen.out.ref
sed -i 's/^\ / /g' /tmp/gen.out.sys
sed -i 's/^\ / /g' /tmp/gen.out.ref
sacrebleu -b -w 2 /tmp/gen.out.ref < /tmp/gen.out.sys
