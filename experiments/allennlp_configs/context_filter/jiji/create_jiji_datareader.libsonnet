function(trans_mode,
         class_mode,
         source_only=false,
         cache_directory=null,
         bert_name='bert-base-uncased',
         window_size=5,
         source_lang='en',
         target_lang='ja',
         context_sentence_index_file=null) {
  type: 'jiji',
  window_size: window_size, 
  translation_data_mode: trans_mode,
  classification_data_mode: class_mode,
  source_max_sequence_length: 512,
  target_max_sequence_length: 512,
  source_lang: source_lang,
  target_lang: target_lang,
  source_only: source_only,
  source_token_indexers: {
    transformer: {
      type: 'customized_pretrained_transformer',
      model_name: bert_name,
    },
  },
  source_tokenizer: {
    type: 'pretrained_transformer',
    model_name: bert_name,
  },
  target_tokenizer: {
    type: 'sentencepiece',
    model_path: '/data/10/litong/NICT-MT/all-4-sentencepiece-en_ja-32000.model',
  },
  cache_directory: cache_directory,
  context_sentence_index_file: context_sentence_index_file,
}
