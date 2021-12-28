import numpy as np

from src.inference.dpmeans_quick import DPMeans


dpmeans = DPMeans(
    max_distance_param=0.1,
    init='dp-means')

# Some random data
X = np.eye(57)[np.random.choice(np.arange(57), replace=True, size=100), :]

dpmeans.fit(X=X)

print(10)
