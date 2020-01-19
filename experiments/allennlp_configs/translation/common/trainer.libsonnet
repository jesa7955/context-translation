//function(cuda_devices=[0, 1]) {
{
  //cuda_device: cuda_device,
  num_epochs: 1000,
  optimizer: {
    lr: 0.0001,
    type: 'adam',
  },
  patience: 10,
  learning_rate_scheduler: {
    type: 'reduce_on_plateau',
    factor: 0.7,
    patience: 8,
  },
  //learning_rate_scheduler: {
  //  type: 'noam',
  //  model_size: 512,
  //  warmup_steps: 8000,
  //},
  //validation_metric: '+BLEU',
}
