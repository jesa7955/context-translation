{
   "dataset_reader": {
      "type": "seq2seq",
      "source_tokenizer": {
          "type": "spacy",
          "language": "en_core_web_sm"
      },
      "target_tokenizer": {
          "type": "spacy",
          "language": "de_core_news_sm"
      },
      "source_max_tokens": 512,
      "target_max_tokens": 512 
   },
   "vocabulary": {
      "tokens_to_add": {
         "source_tokens": ["@start@", "@end@", "@concat"],
         "target_tokens": ["@start@", "@end@", "@concat"]
      }
   },
   "train_data_path": "/data/10/litong/europarl-v9/de-en-train.tsv",
   "validation_data_path": "/data/10/litong/europarl-v9/de-en-dev.tsv",
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
      "type": "composed_seq2seq",
      "source_text_embedder": {
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
      "decoder": {
         "beam_size": 5,
         "tensor_based_metric": {
            "type": "bleu"
         },
         "decoder_net": {
            "type": "lstm_cell",
            "attention": {
               "type": "dot_product"
            },
            "decoding_dim": 500,
            "target_embedding_dim": 500,
         },
         "max_decoding_steps": 50,
         // "scheduled_sampling_ratio": 0.9,
         "target_embedder": {
            "embedding_dim": 500,
            "vocab_namespace": "target_tokens"
         },
         "target_namespace": "target_tokens"
      }
   },
   "trainer": {
      "cuda_device": 3,
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
