local create_dataset_reader(sample_proportion) = {
  "type": "opensubtitles_dataset_reader",
  "window_size": 6,
  "context_size": 3,
  "source_only": true,
  "quality_aware": false,
  "sample_proportion": sample_proportion,
  "source_token_indexers": {
     "bert": {
        "type": "pretrained_transformer",
        "model_name": "bert-base-uncased"
     }     
  },
  "source_tokenizer": {
    "type": "pretrained_transformer",
    "model_name": "bert-base-uncased"
  }

};
{
   "dataset_reader": create_dataset_reader(0.01),
   "validation_dataset_reader": create_dataset_reader(1.0),
   "iterator": {
      "batch_size": 64,
      "sorting_keys": [
         [
            "tokens",
            "num_tokens"
         ]
      ],
      "type": "bucket"
   },
   "model": {
      "type": "pretrained_transformer_for_classification",
      "model_name": "bert-base-uncased"
   },
   //+----------+-------+-------+
   //+ Train    +  Dev  +  Test +
   //+----------+-------+-------+
   //+ 37244708 + 9983  +  9470 +
   //+----------+-------+-------+
   "train_data_path": "/data/10/litong/opensubtitles-v2016-en-fr/train/",
   "validation_data_path": "/data/10/litong/opensubtitles-v2016-en-fr/dev/",
   "trainer": {
      // "type": "callback",
      "cuda_device": [0, 3],
      // "grad_norm": 10,
      "num_epochs": 3,
      "num_serialized_models_to_keep": 1,
      "optimizer": {
         "type": "bert_adam",
         "lr": 2e-5
      },
      "learning_rate_scheduler": {
          "type": "slanted_triangular",
          "num_epochs": 3,
          "num_steps_per_epoch": 120000
      },
      "patience": 5,
      "validation_metric": "+accuracy"
      // "callbacks": [
      //     {"type": "log_to_tensorboard", "log_batch_size_period": 10}
      // ]
   },
}
