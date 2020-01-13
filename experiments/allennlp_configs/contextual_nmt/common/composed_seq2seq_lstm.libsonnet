{
  type: 'composed_seq2seq',
  source_text_embedder: {
    token_embedders: {
      tokens: {
        type: 'embedding',
        embedding_dim: 500,
        trainable: true,
        vocab_namespace: 'source_tokens',
      },
    },
  },
  encoder: {
    type: 'stacked_lstm',
    num_layers: 2,
    hidden_size: 500,
    input_size: 500,
    dropout: 0.1,
  },
  decoder: {
    type: 'bleu_auto_regressive_seq_decoder',
    bleu_exclude_tokens: ['@start@', '@end@', '@concat@'],
    beam_size: 5,
    tensor_based_metric: {
      type: 'bleu',
    },
    decoder_net: {
      type: 'stacked_lstm_cell',
      attention: {
        type: 'additive',
        vector_dim: 500,
        matrix_dim: 500,
      },
      layer_num: 2,
      hidden_dim: 500,
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
}
