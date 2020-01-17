function (trans_mode, class_mode, source_only=false) {
  type: 'jiji',
  window_size: 1,
  context_size: 1,
  translation_data_mode: trans_mode,
  classification_data_mode: class_mode,
  source_max_sequence_length: 512,
  target_max_sequence_length: 512,
  source_only: source_only,
  source_token_indexers: {
    transformer: {
      type: 'pretrained_transformer_customized',
      model_name: 'bert-base-uncased',
    },
  },
  source_tokenizer: {
    type: 'pretrained_transformer',
    model_name: 'bert-base-uncased',
  },
  target_tokenizer: {
    type: 'sentencepiece',
    model_path: '/data/10/litong/jiji-with-document-boundaries/jiji_sentencepiece_ja.model',
  },
}
