local create_configuration = import '../create_configuration.libsonnet';
local create_dataset_reader = import '../create_jiji_datareader.libsonnet';

local dataset_reader = create_dataset_reader('2-to-1',
                                             'train',
                                             false,
                                             context_indicators_file='/home/litong/context_translation/resources/all_en_ja_jparacrawl_base_context_indicators_fa5bdc4816e1f28d25b99b1227af001b.pkl',
                                             context_indicators_key='nll_loss');

create_configuration(dataset_reader, cuda_device=0, batch_size=48)
