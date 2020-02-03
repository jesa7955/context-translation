local create_dataset_reader = import '../create_jiji_datareader.libsonnet';
local create_configuration = import '../create_configuration_nodecoder.libsonnet';

local dataset_reader = create_dataset_reader("2-to-1",
                                             "inference",
                                              true);

create_configuration(dataset_reader)
