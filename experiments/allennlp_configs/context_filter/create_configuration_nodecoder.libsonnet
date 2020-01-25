local num_epochs = 2;

function(dataset_reader,
         bert_name='bert-base-uncased',
         batch_size=64,
         cuda_device=0) {
  dataset_reader: dataset_reader,
  iterator: {
    batch_size: batch_size,
    type: 'bucket',
  },
  model: {
    type: 'context_sentence_filter',
    model_name: bert_name,
    num_labels: 2,
    load_classifier: true,
    transformer_trainable: true,
    classifier_traninable: true,
  },
  train_data_path: '/home/litong/context_translation/resources/train_77b9dbd0538187438b8dd13a8f6b935c.pkl',
  validation_data_path: '/home/litong/context_translation/resources/valid_0a06896723176aff827aac15a2e1ac94.pkl',
  test_data_path: '/home/litong/context_translation/resources/test_3c5cba7d2f64b7fb2d6b5013adc7a888.pkl',
  //train_data_path: '/home/litong/context_translation/resources/valid_0a06896723176aff827aac15a2e1ac94.pkl',
  //validation_data_path: '/home/litong/context_translation/resources/test_3c5cba7d2f64b7fb2d6b5013adc7a888.pkl',
  trainer: {
    cuda_device: cuda_device,
    num_epochs: num_epochs,
    num_serialized_models_to_keep: 1,
    optimizer: {
      type: 'adamw',
      lr: 5e-5,
    },
    learning_rate_scheduler: {
      type: 'slanted_triangular',
      num_epochs: num_epochs,
      num_steps_per_epoch: 22000,
    },
    patience: 5,
    validation_metric: '+accuracy',
  },
}
