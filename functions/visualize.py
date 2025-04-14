import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
import numpy as np

def cluster_plot(optimal_k, clusters, X_pca, fname):
    """
    Visualize the clustering results by plotting the clusters and their centroids.
    """ 
    # Define a color palette with enough distinct colors for your clusters.
    # Seaborn's color palette is handy for this.
    palette = sns.color_palette("bright", n_colors=optimal_k)

    plt.figure(figsize=(8, 6))

    for cluster_id in range(optimal_k):
        # Extract the points in this cluster
        cluster_points = X_pca[clusters == cluster_id]
        
        # Plot the individual data points
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=40,
            color=palette[cluster_id],
            label=f"Cluster {cluster_id + 1}",
            alpha=0.7
        )
        
        # Compute and draw the convex hull, if there are enough points
        if len(cluster_points) > 2:
            hull = ConvexHull(cluster_points)
            # Create a polygon by connecting the hull vertices
            hull_vertices = np.append(hull.vertices, hull.vertices[0])  # close the polygon
            plt.fill(
                cluster_points[hull_vertices, 0],
                cluster_points[hull_vertices, 1],
                color=palette[cluster_id],
                alpha=0.2
            )

    plt.title("K-means Clusters with Convex Hull Boundaries (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()

    # Save the plot to a file
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {fname}")

    plt.show()

def cluster_bar_charts(cluster_top_packages, optimal_k, fname):
    """
    Create a bar chart for each cluster showing the top 10 packages,
    using the same bright color palette as your cluster_plot().
    
    Each chart is saved to disk with a filename based on the provided fname.
    
    Parameters:
       cluster_top_packages (dict): A dictionary mapping cluster indices (0,1,...)
                                    to a list of tuples (package, frequency).
       optimal_k (int): Total number of clusters.
       fname (str): Base filename/path for saving out the charts.
    """
    # Use the same bright color palette
    palette = sns.color_palette("bright", n_colors=optimal_k)
    
    for cluster in sorted(cluster_top_packages.keys()):
        top10 = cluster_top_packages[cluster]
        if not top10:
            continue
        # Unpack the package names and frequencies
        packages, frequencies = zip(*top10)
        
        plt.figure(figsize=(8,6))
        # Use the color corresponding to this cluster
        plt.bar(packages, frequencies, color=palette[cluster])
        plt.title(f"Top 10 Packages in Cluster {cluster + 1}")
        plt.xlabel("Packages")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Build a unique filename for this cluster and save the plot
        filename = f"{fname}_cluster{cluster + 1}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved bar chart for Cluster {cluster + 1} as {filename}")
        
        plt.show()
        plt.close()  # Good practice to close the figure after saving/showing.
