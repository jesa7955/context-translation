local model = import '../common/composed_seq2seq_lstm.libsonnet';
local create_trainer = import '../common/trainer.libsonnet';
local create_configuration = import 'create_configuration.libsonnet';
local create_dataset_reader = import 'dataset_reader.libsonnet';

create_configuration(create_dataset_reader(0, 0), create_trainer(2), model)
