function(dataset_reader, trainer, model) {
  dataset_reader: dataset_reader,
  trainer: trainer,
  model: model,
  train_data_path: '/data/10/litong/europarl-v7/europarl-v7.de-en.train',
  validation_data_path: '/data/10/litong/europarl-v7/europarl-v7.de-en.dev',
  vocabulary: {
    tokens_to_add: {
      source_tokens: ['@start@', '@end@', '@concat'],
      target_tokens: ['@start@', '@end@', '@concat'],
    },
  },
  iterator: {
    batch_size: 64,
    sorting_keys: [
      [
        'source_tokens',
        'num_tokens',
      ],
    ],
    type: 'bucket',
  },
}
