import pandas as pd  
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import gc
os.chdir('C:/Kaige_Research/Code/graph_bandit/code/')
input_path='../original_data/jester/'
output_path='../processed_data/jester/'

import pandas as pd  
import numpy as np
import json
import matplotlib.pyplot as plt
import os
os.chdir('C:/Kaige_Research/Code/graph_bandit/code/')
input_path='../original_data/jester/'
output_path='../processed_data/jester/'
print(os.listdir(input_path))

rating=pd.read_csv(input_path+'jester-data-1.xls', delimiter="\t")