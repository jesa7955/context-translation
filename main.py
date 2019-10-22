import luigi
import gokart
import numpy as np

import context_nmt

if __name__ == '__main__':
    luigi.configuration.LuigiConfigParser.add_config_path('./conf/param.ini')
    np.random.seed(57)
    gokart.run()
