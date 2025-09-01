from envs.jamming_dataset_environment import JammingDatasetEnvironment
import pandas as pd, numpy as np

# Create synthetic dataset
N=200
labels=['normal']*(N//2)+['power_jamming']*(N//2)
import random
random.shuffle(labels)
df=pd.DataFrame({
    'label': labels,
    'f1': np.random.randn(N)*0.5 + (np.array(labels)!="normal")*0.3,
    'f2': np.random.randn(N)*0.3 + 2 + (np.array(labels)!="normal")*0.2,
    'f3': np.random.randn(N),
    'f4': np.random.randn(N)*0.1 + 5
})
df.to_csv('synthetic_dataset.csv', index=False)

env=JammingDatasetEnvironment('synthetic_dataset.csv')
state,_=env.reset()
print('State dim', state.shape)
import numpy as np
Action=np.zeros(5)
ret=0
for step in range(30):
    ns,r,done,info=env.step(Action)
    ret+=r
    if done:
        break
print('Return', ret, 'steps', step+1)
print('Detection rate metric', env.get_performance_metrics())
