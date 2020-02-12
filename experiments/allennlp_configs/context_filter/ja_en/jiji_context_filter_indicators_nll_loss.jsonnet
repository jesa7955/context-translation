local create_configuration = import '../create_configuration.libsonnet';
local create_dataset_reader = import '../create_jiji_datareader.libsonnet';
local bert_name = 'bert-base-japanese';

local dataset_reader = create_dataset_reader('2-to-1',
                                             'train',
                                             false,
                                             bert_name=bert_name,
                                             source_lang='ja',
                                             target_lang='en',
                                             context_indicators_file='/home/litong/context_translation/resources/all_ja_en_jparacrawl_base_context_indicators_02eae9e102854720cad92024515cc735.pkl',
                                             context_indicators_key='nll_loss');

create_configuration(dataset_reader, bert_name=bert_name, batch_size=48, cuda_device=1)
