local model = import '../common/composed_seq2seq_transformer.libsonnet';
local create_trainer = import '../common/trainer_single.libsonnet';
local create_configuration = import 'create_configuration.libsonnet';
local create_dataset_reader = import 'dataset_reader.libsonnet';
local trans_tokenizer = {
  type: 'pretrained_transformer',
  model_name: 'bert-base-uncased',
};
local en_tokenizer = {
  type: 'sentencepiece',
  model_path: '/data/10/litong/jiji-with-document-boundaries/jiji_sentencepiece_en.model',
};
local ja_tokenizer = {
  type: 'sentencepiece',
  model_path: '/data/10/litong/jiji-with-document-boundaries/jiji_sentencepiece_ja.model',
};
local share_tokenizer = {
  type: 'sentencepiece',
  model_path: '/data/10/litong/jiji-with-document-boundaries/jiji_sentencepiece_en_ja.model',
};
local common_token_indexers = {
  tokens: {
    type: 'single_id',
    namespace: 'tokens',
  },
};
local train_dataset_reader = create_dataset_reader('2-to-1', 'none', share_tokenizer, share_tokenizer, common_token_indexers, common_token_indexers, true);
local val_dataset_reader = create_dataset_reader('2-to-1', 'none', share_tokenizer, share_tokenizer, common_token_indexers, common_token_indexers, true);

create_configuration(train_dataset_reader, val_dataset_reader, create_trainer(1), model)
