{
  type: 'simple_seq2seq',
  source_embedder: {
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
  max_decoding_steps: 85,
  target_namespace: 'target_tokens',
  target_embedding_dim: 500,
  attention: {
    type: 'additive',
    vector_dim: 500,
    matrix_dim: 500,
  },
  beam_size: 5,
  use_bleu: true,
}
