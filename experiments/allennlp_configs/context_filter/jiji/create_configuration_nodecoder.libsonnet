local num_epochs = 2;

function(dataset_reader,
         bert_name='bert-base-uncased') {
  dataset_reader: dataset_reader,
  iterator: {
    batch_size: 64,
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
  train_data_path: '/home/litong/context_translation/resources/train_aeabd4a49ce40d291d72fb22e4f84f70.pkl',
  validation_data_path: '/home/litong/context_translation/resources/valid_93dd7f59179d59481dffa319f8eceadb.pkl',
  test_data_path: '/home/litong/context_translation/resources/test_3c5cba7d2f64b7fb2d6b5013adc7a888.pkl',
  //train_data_path: "/home/litong/context_translation/resources/valid_93dd7f59179d59481dffa319f8eceadb.pkl",
  //validation_data_path: "/home/litong/context_translation/resources/test_3c5cba7d2f64b7fb2d6b5013adc7a888.pkl",
  trainer: {
    cuda_device: 1,
    num_epochs: num_epochs,
    num_serialized_models_to_keep: 1,
    optimizer: {
      type: 'adamw',
      lr: 5e-5,
    },
    learning_rate_scheduler: {
      type: 'slanted_triangular',
      num_epochs: num_epochs,
      num_steps_per_epoch: 21427,
    },
    patience: 5,
    validation_metric: '+accuracy',
  },
}
