local create_dataset_reader = import 'create_jiji_datareader.libsonnet';

{
  dataset_reader: create_dataset_reader('2-to-1', 'train'),
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
    seq_decoder: {
      decoder_net: {
        type: 'lstm_cell',
        attention: {
          type: 'fixed',
        },
        decoding_dim: 768,
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
  train_data_path: '/data/10/litong/jiji-with-document-boundaries/train.json',
  validation_data_path: '/data/10/litong/jiji-with-document-boundaries/dev.json',
  test_data_path: '/data/10/litong/jiji-with-document-boundaries/test.json',
  //train_data_path: '/data/10/litong/jiji-with-document-boundaries/dev.json',
  //validation_data_path: '/data/10/litong/jiji-with-document-boundaries/test.json',
  //distributed: {
  //  cuda_devices: [0, 1],
  //},
  trainer: {
    cuda_device: 0,
    num_epochs: 1,
    num_serialized_models_to_keep: 1,
    optimizer: {
      type: 'bert_adam',
      lr: 5e-5,
      parameter_groups: [[
        [
          'decoder_cell.weight_ih',
          'decoder_cell.bias_ih',
          'decoder_cell.weight_hh',
          'decoder_cell.bias_hh',
          'output_projection_layer.weight',
          'output_projection_layer.bias',
        ],
        { lr: 0.001 },
      ]],
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
