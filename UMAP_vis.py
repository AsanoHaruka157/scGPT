import pandas as pd
import umap.umap_ as umap
import matplotlib.pyplot as plt

# Load the embeddings and definitions
embeddings_path = './data/phecode_embeddings.csv'
definitions_path = './data/phecode_definitions1.2.csv'

emb = pd.read_csv(embeddings_path)
emb = emb.rename(columns={'Unnamed: 0': 'phecode'})
defs = pd.read_csv(definitions_path)

merged = emb.merge(defs, on='phecode', how='left')
# Extract embedding columns '0' through '199'
feature_cols = [str(i) for i in range(200)]
X = merged[feature_cols].values

# Run UMAP to reduce to 2 dimensions
reducer = umap.UMAP(n_neighbors=5, n_components=2, metric='cosine', min_dist=0.2)
X2 = reducer.fit_transform(X)

for cat in merged['category'].unique():
    mask = (merged['category'] == cat)
    plt.scatter(X2[mask, 0], X2[mask, 1], label=cat, s=15, alpha=0.7)

# Plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.title('UMAP Projection of Phecode Embeddings by Category')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.tight_layout()
plt.show()