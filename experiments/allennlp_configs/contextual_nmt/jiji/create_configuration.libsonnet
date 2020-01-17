function(dataset_reader, val_dataset_reader, trainer, model, batch_size=1024, val_interval=4000) {
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
      tokens: ['@concat@'],
    },
  },
  iterator: {
    batch_size: 256,
    sorting_keys: [
      [
        'source_tokens',
        'num_tokens',
      ],
    ],
    maximum_samples_per_batch: ['num_tokens', batch_size],
    instances_per_epoch: val_interval,
    cache_instances: true,
    type: 'bucket',
  },
}
