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
  train_data_path: '/home/litong/context_translation/resources/train_d84236a6b7a9cb804e0de766ced5aa1d.pkl',
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
      num_epochs: num_epochs,
      num_steps_per_epoch: 22000,
    },
    patience: 5,
    validation_metric: '+accuracy',
  },
}
