local create_dataset_reader(ws=6, cs=3) = {
  "type": "jiji",
  "window_size": ws,
  "context_size": cs,
  "source_only": true,
  "source_max_sequence_length": 512,
  "target_max_sequence_length": 512,
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
   "iterator": {
      "batch_size": 24,
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
   "validation_data_path": "/data/10/litong/jiji-with-document-boundaries/dev.json",
   "test_data_path": "/data/10/litong/jiji-with-document-boundaries/test.json",
   "trainer": {
      "cuda_device": [0, 1, 3],
      "num_epochs": 3,
      "num_serialized_models_to_keep": 1,
      "optimizer": {
         "type": "bert_adam",
         "lr": 5e-5
      },
      "learning_rate_scheduler": {
          "type": "slanted_triangular",
          "num_epochs": 3,
          "num_steps_per_epoch": 16000
      },
      "patience": 5,
      "validation_metric": "+accuracy"
   },
}
