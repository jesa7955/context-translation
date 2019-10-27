function(cuda_device=3) {
  cuda_device: cuda_device,
  num_epochs: 100,
  num_serialized_models_to_keep: 1,
  optimizer: {
    lr: 0.001,
    type: 'adam',
  },
  patience: 10,
  learning_rate_scheduler: {
    type: 'reduce_on_plateau',
    mode: 'max',
    factor: 0.5,
    patience: 5,
  },
  validation_metric: '+BLEU',
}
