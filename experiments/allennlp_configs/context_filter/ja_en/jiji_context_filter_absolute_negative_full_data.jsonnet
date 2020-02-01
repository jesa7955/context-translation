local create_configuration = import '../create_configuration_full_data.libsonnet';
local create_dataset_reader = import '../create_jiji_datareader.libsonnet';
local bert_name = 'bert-base-japanese';

local dataset_reader = create_dataset_reader('2-to-1',
                                             'train',
                                             false,
                                             bert_name=bert_name,
                                             source_lang='ja',
                                             target_lang='en',
                                             noisy_context_dataset_path='/home/litong/context_translation/resources/jparacrawl_0.77.pkl');

create_configuration(dataset_reader, bert_name=bert_name, batch_size=48, cuda_device=1)
