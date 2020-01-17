function(trans_mode, class_mode, src_tokenizer, tgt_tokenizer, src_indexers={}, tgt_indexers={}, share_vocabulary=false, source_add_factors=false) {
  type: 'jiji',
  context_size: 1,
  window_size: 1,
  source_max_sequence_length: 128,
  target_max_sequence_length: 128,
  source_vocabulary_size: 16000,
  target_vocabulary_size: 16000,
  score_threshold: 0.3,
  translation_data_mode: trans_mode,
  classification_data_mode: class_mode,
  source_token_indexers: src_indexers,
  target_token_indexers: tgt_indexers,
  // source_token_indexers: {
  //   transformer: {
  //     type: 'pretrained_transformer_customized',
  //     model_name: 'bert-base-uncased',
  //   },
  // },
  source_tokenizer: src_tokenizer,
  target_tokenizer: tgt_tokenizer,
  share_vocabulary: share_vocabulary,
  source_add_factors: source_add_factors,
}
