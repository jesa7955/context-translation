local create_dataset_reader = import '../create_jiji_datareader.libsonnet';
local create_configuration = import '../create_configuration_nodecoder.libsonnet';
local bert_name = "bert-base-japanese";

local dataset_reader = create_dataset_reader('2-to-1', 'train', true, bert_name=bert_name, source_lang='ja', target_lang='en');

create_configuration(dataset_reader, bert_name=bert_name, cuda_device=0)
