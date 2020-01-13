local create_dataset_reader(trans_mode, class_mode) = {
  type: 'jiji',
  window_size: 1,
  context_size: 1,
  translation_data_mode: trans_mode,
  classification_data_mode: class_mode,
  source_max_sequence_length: 85,
  target_max_sequence_length: 85,
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
};
{
  dataset_reader: create_dataset_reader("2-to-1", "train"),
  // validation_dataset_reader: create_dataset_reader("2-to-1", "inference"),
  iterator: {
    batch_size: 128,
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
  train_data_path: '/data/10/litong/jiji-with-document-boundaries/dev.json',
  validation_data_path: '/data/10/litong/jiji-with-document-boundaries/test.json',
  distributed: {
    cuda_devices: [0, 1],
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
