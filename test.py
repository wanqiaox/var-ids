import dask
import time
import random
import pandas as pd
import numpy as np

def costly_simulation(list_param):
    time.sleep(random.random())
    return [sum(list_param)], [sum(list_param)/len(list_param)]

input_params = pd.DataFrame(np.random.random(size=(500, 4)),
                            columns=['param_a', 'param_b', 'param_c', 'param_d'])

results = []

for parameters in input_params.values[:10]:
    lazy_result_tuple = dask.delayed(costly_simulation)(parameters)
    results.append(lazy_result_tuple)

results = dask.compute(*results)

print(list(zip(*results)))
pass