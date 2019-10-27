function(source_max_tokens=96, target_max_tokens=96) {
  type: 'seq2seq',
  source_max_tokens: source_max_tokens,
  target_max_tokens: target_max_tokens,
  source_tokenizer: {
    type: 'sentencepiece',
    model_path: '/data/10/litong/europarl-v7/europarl-v7.en.sp.model',
  },
  target_tokenizer: {
    type: 'sentencepiece',
    model_path: '/data/10/litong/europarl-v7/europarl-v7.de.sp.model',
  },
}
