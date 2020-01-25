local create_dataset_reader = import '../create_jiji_datareader.libsonnet';
local create_configuration = import '../create_configuration.libsonnet';

local dataset_reader = create_dataset_reader("2-to-1",
                                             "inference",
                                              true,
                                              window_size=5);
                                              //cache_directory='/data/10/litong/context_filter_infer_cache_en_ja_ws5');

create_configuration(dataset_reader)
