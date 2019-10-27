{
   "dataset_reader": {
      "type": "jiji_dataset_reader",
      "context_size": 0,
      "window_size": 0,
      "score_threshold": 0.3,
      "source_max_sequence_length": 85,
      "target_max_sequence_length": 85,
      "quality_aware": true,
      "source_tokenizer": {
          "type": "sentencepiece",
          "model_path": "/data/10/litong/jiji-with-document-boundaries/jiji_sentencepiece_en.model"
      },
      "target_tokenizer": {
          "type": "sentencepiece",
          "model_path": "/data/10/litong/jiji-with-document-boundaries/jiji_sentencepiece_ja.model"
      }
   },
   "vocabulary": {
      "tokens_to_add": {
         "source_tokens": ["@start@", "@end@", "@concat"],
         "target_tokens": ["@start@", "@end@", "@concat"]
      }
   },
   "train_data_path": "/data/10/litong/jiji-with-document-boundaries/train.json",
   "validation_data_path": "/data/10/litong/jiji-with-document-boundaries/dev.json",
   "test_data_path": "/data/10/litong/jiji-with-document-boundaries/test.json",
   "iterator": {
      "batch_size": 64,
      "sorting_keys": [
         [
            "source_tokens",
            "num_tokens"
         ]
      ],
      "type": "bucket"
   },
   "model": {
      "type": "simple_seq2seq",
      "source_embedder": {
         "token_embedders": {
            "tokens": {
               "type": "embedding",
               "embedding_dim": 500,
               "trainable": true,
               "vocab_namespace": "source_tokens"
            }
         }
      },
      "encoder": {
         "type": "lstm",
         "num_layers": 2,
         "hidden_size": 500,
         "input_size": 500,
         "dropout": 0.1
      },
      "max_decoding_steps": 85,
      "target_namespace": "target_tokens",
      "target_embedding_dim": 500,
      "attention": {
         "type": "additive",
         "vector_dim": 500,
         "matrix_dim": 500
      },
      "beam_size": 5,
      "use_bleu": true
   },
   "trainer": {
      "cuda_device": 2,
      "num_epochs": 100,
      "num_serialized_models_to_keep": 1,
      "optimizer": {
         "lr": 0.001,
         "type": "adam"
      },
      "patience": 10,
      "learning_rate_scheduler": {
          "type": "reduce_on_plateau",
          "mode": "max",
          "factor": 0.5,
          "patience": 5,
      },
      "validation_metric": "+BLEU"
   },
}
