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
    type: 'lstm',
    num_layers: 2,
    hidden_size: 500,
    input_size: 500,
    dropout: 0.1,
  },
  decoder: {
    beam_size: 5,
    tensor_based_metric: {
      type: 'bleu',
    },
    decoder_net: {
      type: 'lstm_cell',
      attention: {
        type: 'dot_product',
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
}
