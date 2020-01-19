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
  train_data_path: '/home/litong/context_translation/resources/train_aeabd4a49ce40d291d72fb22e4f84f70.pkl',
  validation_data_path: '/home/litong/context_translation/resources/valid_93dd7f59179d59481dffa319f8eceadb.pkl',
  test_data_path: '/home/litong/context_translation/resources/test_3c5cba7d2f64b7fb2d6b5013adc7a888.pkl',
  //train_data_path: '/home/litong/context_translation/resources/valid_93dd7f59179d59481dffa319f8eceadb.pkl',
  //validation_data_path: '/home/litong/context_translation/resources/test_3c5cba7d2f64b7fb2d6b5013adc7a888.pkl',
  trainer: {
    cuda_device: 0,
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
      num_steps_per_epoch: 21427,
    },
    patience: 5,
    validation_metric: '+accuracy',
  },
}
