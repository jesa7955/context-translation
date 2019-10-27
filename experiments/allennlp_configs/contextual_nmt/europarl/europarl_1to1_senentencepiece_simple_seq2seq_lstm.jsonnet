local model = import '../common/simple_seq2seq_lstm.libsonnet';
local create_trainer = import '../common/trainer.libsonnet';
local create_configuration = import 'create_configuration.libsonnet';
local create_dataset_reader = import 'dataset_reader.libsonnet';

create_configuration(create_dataset_reader(), create_trainer(3), model)
