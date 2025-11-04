import pandas as pd
from structure import Structure

class DimensionalityReduction(Structure):
    
    def apply_pca(X: pd.DataFrame, n_components: float | int) -> pd.DataFrame:
        # Dimensionality Reduction
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)
        return X
    
    def apply_tsne(X: pd.DataFrame, n_components: int = 3, perplexity: int = 30, random_state: int = 42) -> pd.DataFrame:
        # Dimensionality Reduction
        from sklearn.manifold import TSNE
        
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        X = tsne.fit_transform(X)
        return X