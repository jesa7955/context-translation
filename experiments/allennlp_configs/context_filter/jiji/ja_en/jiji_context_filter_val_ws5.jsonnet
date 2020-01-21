local create_configuration = import '../create_configuration.libsonnet';
local create_dataset_reader = import '../create_jiji_datareader.libsonnet';
local bert_name = 'bert-base-japanese';

local dataset_reader = create_dataset_reader('2-to-1',
                                             'inference',
                                             true,
                                             bert_name=bert_name,
                                             source_lang='ja',
                                             target_lang='en',
                                             window_size=5);

create_configuration(dataset_reader, bert_name=bert_name)
