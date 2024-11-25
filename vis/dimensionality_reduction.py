import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load the digits dataset (8x8 images)
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Normalize the data
    X_normalized = (X - X.mean()) / X.std()

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_normalized)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_normalized)

    # Create figure with two subplots
    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot PCA results
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=y,
        palette='deep',
        edgecolor='none',
        ax=ax1
    )
    ax1.set_title('PCA visualization of MNIST (8x8)')
    ax1.set_xlabel('First Principal Component')
    ax1.set_ylabel('Second Principal Component')

    # Plot t-SNE results
    sns.scatterplot(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        hue=y,
        palette='deep',
        edgecolor='none',
        ax=ax2
    )
    ax2.set_title('t-SNE visualization of MNIST (8x8)')
    ax2.set_xlabel('First t-SNE Component')
    ax2.set_ylabel('Second t-SNE Component')

    plt.tight_layout()

    # Save the figure instead of showing
    plt.savefig('./data/dimensionality_reduction.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Print explained variance ratio for PCA
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

if __name__ == "__main__":
    main()
