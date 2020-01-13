local model = import '../common/composed_seq2seq_transformer.libsonnet';
local create_trainer = import '../common/trainer_noam_single.libsonnet';
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
local train_dataset_reader = create_dataset_reader('1-to-1', 'none', {}, en_tokenizer, ja_tokenizer);
local val_dataset_reader = create_dataset_reader('1-to-1', 'none', {}, en_tokenizer, ja_tokenizer);

create_configuration(train_dataset_reader, val_dataset_reader, create_trainer(0), model)
