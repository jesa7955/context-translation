local create_dataset_reader(ws=1, cs=1) = {
  type: 'jiji',
  window_size: ws,
  context_size: cs,
  source_only: true,
  source_max_sequence_length: 512,
  target_max_sequence_length: 512,
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

};
{
  dataset_reader: create_dataset_reader(),
  iterator: {
    batch_size: 24,
    sorting_keys: [
      [
        'tokens',
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
    seq_decoder: {
      decoder_net: {
        type: 'lstm_cell',
        attention: {
          type: 'fixed',
        },
        decoding_dim: 500,
        target_embedding_dim: 500,
      },
      max_decoding_steps: 85,
      target_embedder: {
        embedding_dim: 500,
        vocab_namespace: 'target_tokens',
      },
      target_namespace: 'target_tokens',
    },
  },
  //"train_data_path": "/data/10/litong/jiji-with-document-boundaries/train.json",
  //"validation_data_path": "/data/10/litong/jiji-with-document-boundaries/dev.json",
  //"test_data_path": "/data/10/litong/jiji-with-document-boundaries/test.json",
  train_data_path: '/data/10/litong/jiji-with-document-boundaries/toy_train.json',
  validation_data_path: '/data/10/litong/jiji-with-document-boundaries/toy_test.json',
  distributed: {
    cuda_devices: [0, 1, 2, 3],
  },
  trainer: {
    num_epochs: 3,
    num_serialized_models_to_keep: 1,
    optimizer: {
      type: 'bert_adam',
      lr: 5e-5,
    },
    learning_rate_scheduler: {
      type: 'slanted_triangular',
      num_epochs: 3,
      num_steps_per_epoch: 16000,
    },
    patience: 5,
    validation_metric: '+accuracy',
  },
}
