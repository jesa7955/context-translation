local create_configuration = import '../create_configuration.libsonnet';
local create_dataset_reader = import '../create_jiji_datareader.libsonnet';

local dataset_reader = create_dataset_reader('2-to-1',
                                             'train',
                                             false,
                                             noisy_context_dataset_path='/home/litong/context_translation/resources/jparacrawl_0.77.pkl');

create_configuration(dataset_reader, cuda_device=0, batch_size=48)
