local create_dataset_reader = import 'create_jiji_datareader.libsonnet';

{
  dataset_reader: create_dataset_reader('2-to-1', 'train', true),
  validation_dataset_reader: create_dataset_reader("2-to-1", "inference"),
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
  model: {
    type: 'context_sentence_filter',
    model_name: 'bert-base-uncased',
    num_labels: 2,
    load_classifier: true,
    transformer_trainable: true,
    classifier_traninable: true,
  },
  train_data_path: '/data/10/litong/jiji-with-document-boundaries/train.json',
  validation_data_path: '/data/10/litong/jiji-with-document-boundaries/dev.json',
  test_data_path: '/data/10/litong/jiji-with-document-boundaries/test.json',
  //train_data_path: '/data/10/litong/jiji-with-document-boundaries/dev.json',
  //validation_data_path: '/data/10/litong/jiji-with-document-boundaries/test.json',
  //distributed: {
  //  cuda_devices: [0, 1],
  //},
  trainer: {
    cuda_device: 1,
    num_epochs: 1,
    num_serialized_models_to_keep: 1,
    optimizer: {
      type: 'bert_adam',
      lr: 5e-5,
    },
    learning_rate_scheduler: {
      type: 'slanted_triangular',
      num_epochs: 1,
      num_steps_per_epoch: 2300,
    },
    patience: 5,
    validation_metric: '+accuracy',
  },
}
