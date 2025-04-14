import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Use elbow method to determine the optimal number of clusters
def elbow_method(X, fname, max_k=10):
    """
    Plots the sum of squared errors (SSE) for K-means with k=1..max_k.
    Helps determine an 'elbow' (a point of diminishing returns) for cluster count.
    """
    sse = []
    k_values = range(1, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)  # .inertia_ is the SSE for K-means
    
    plt.figure(figsize=(6, 4))
    plt.plot(k_values, sse, marker='o')
    plt.xticks(k_values)
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Sum of Squared Errors (SSE)")
    
    # Save the plot to a file
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {fname}")
    
    plt.show()