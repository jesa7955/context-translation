{
  type: 'factored_composed_seq2seq',
  source_text_embedder: {
    token_embedders: {
      tokens: {
        type: 'embedding',
        embedding_dim: 512,
        trainable: true,
        vocab_namespace: 'tokens',
      },
    },
  },
  source_factor_method: 'concat',
  source_factor_embedder: {
    token_embedders: {
      factors: {
        type: 'embedding',
        embedding_dim: 16,
        trainable: true,
        vocab_namespace: 'source_factors',
      },
    },
  },
  encoder: {
    type: 'stacked_self_attention',
    input_dim: 528,
    hidden_dim: 512,
    projection_dim: 512,
    feedforward_hidden_dim: 2048,
    num_layers: 6,
    num_attention_heads: 8,
    use_positional_encoding: true,
    dropout_prob: 0.2,
    residual_dropout_prob: 0.2,
    attention_dropout_prob: 0.2,
  },
  decoder: {
    type: 'bleu_auto_regressive_seq_decoder',
    label_smoothing_ratio: 0.1,
    bleu_exclude_tokens: ['@start@', '@end@', '@concat@'],
    beam_size: 5,
    tensor_based_metric: {
      type: 'bleu',
    },
    decoder_net: {
      type: 'stacked_self_attention',
      decoding_dim: 512,
      target_embedding_dim: 512,
      feedforward_hidden_dim: 2048,
      num_layers: 6,
      num_attention_heads: 8,
      use_positional_encoding: true,
      dropout_prob: 0.2,
      residual_dropout_prob: 0.2,
      attention_dropout_prob: 0.2,
    },
    tie_output_embedding: true,
    max_decoding_steps: 128,
    target_embedder: {
      embedding_dim: 512,
      vocab_namespace: 'tokens',
    },
  },
  tied_source_embedder_key: 'tokens',
  initializer: [
    ['encoder.*', { type: 'xavier_uniform', gain: 3.0 }],
    ['decoder.*', { type: 'xavier_uniform', gain: 3.0 }],
    // [".*embed.*", {"type": "normal", "mean": 0, "std": 23}],
    ['.*norm.*', 'prevent'],
    ['.*bias.*', 'prevent'],
  ],
}
