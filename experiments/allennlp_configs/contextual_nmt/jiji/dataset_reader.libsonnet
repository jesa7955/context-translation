function(context_size=3, window_size=6) {
  type: 'jiji',
  context_size: context_size,
  window_size: window_size,
  source_max_sequence_length: 85,
  target_max_sequence_length: 85,
  quality_aware: true,
  score_threshold: 0.3,
  //source_tokenizer: {
  //  type: 'sentencepiece',
  //  model_path: '/data/10/litong/jiji-with-document-boundaries/jiji_sentencepiece_en.model',
  //},
  //target_tokenizer: {
  //  type: 'sentencepiece',
  //  model_path: '/data/10/litong/jiji-with-document-boundaries/jiji_sentencepiece_ja.model',
  //},
}
