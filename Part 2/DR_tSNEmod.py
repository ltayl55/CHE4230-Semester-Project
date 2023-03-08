#Method 1 for DR t-SNE
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_excel(r"./../Data/RawData.xlsx")
#scaling the data so it can be used in the t-SNE

scale = StandardScaler().fit(df)

df_scaled = scale.transform(df)

#t-SNE code

from sklearn.manifold import TSNE

n_components = 2
tsne = TSNE(n_components)
tsne_result = tsne.fit_transform(df_scaled)
tsne_result.shape