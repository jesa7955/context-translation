local create_dataset_reader(ws=6, cs=3) = {
  "type": "jiji_dataset_reader",
  "window_size": ws,
  "context_size": cs,
  "source_only": true,
  "quality_aware": false,
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
   "dataset_reader": create_dataset_reader(),
   // "validation_dataset_reader": create_dataset_reader(-1, 3),
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
   "train_data_path": "/data/10/litong/jiji-with-document-boundaries/train.json",
   "validataion_data_path": "/data/10/litong/jiji-with-document-boundaries/dev.json",
   "test_data_path": "/data/10/litong/jiji-with-document-boundaries/test.json",
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
