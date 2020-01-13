function(dataset_reader, val_dataset_reader, trainer, model) {
  dataset_reader: dataset_reader,
  validation_dataset_reader: val_dataset_reader,
  trainer: trainer,
  model: model,
  train_data_path: '/data/10/litong/jiji-with-document-boundaries/train.json',
  validation_data_path: '/data/10/litong/jiji-with-document-boundaries/dev.json',
  test_data_path: '/data/10/litong/jiji-with-document-boundaries/test.json',
  //train_data_path: '/data/10/litong/jiji-with-document-boundaries/dev.json',
  //validation_data_path: '/data/10/litong/jiji-with-document-boundaries/test.json',
  vocabulary: {
    tokens_to_add: {
      source_tokens: ['@concat@'],
      target_tokens: ['@concat@'],
    },
  },
  iterator: {
    batch_size: 32,
    sorting_keys: [
      [
        'source_tokens',
        'num_tokens',
      ],
    ],
    type: 'bucket',
  },
}
