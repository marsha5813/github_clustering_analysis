# Libraries
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter
import os
import pickle

# Import my functions
from functions.collect import scrape_github, check_remaining_queries, extract_dependencies
from functions.cluster import elbow_method 
from functions.visualize import cluster_plot, cluster_bar_charts

# Note that I saved my GitHub API token to an environment variable
token = os.environ.get("GITHUB_TOKEN")

# See how many queries I have left in my GitHub API rate limit window
check_remaining_queries(token)

# Scrape github repositories for package data
raw = scrape_github(token = token, max_repos=4000, min_stars=100)

# Extract and deduplicate dependencies from the scraped repositories
repos = extract_dependencies(raw)

# You can avoid re-running the scrape and extract functions by loading the pickled objects
with open("data/repos.pkl", "rb") as infile:
    repos = pickle.load(infile)

# Check results
for repo_name, deps in repos.items():
    print(f"{repo_name}: {deps}")

# Save objects to disk using pickle
with open("data/raw.pkl", "wb") as outfile:
    pickle.dump(raw, outfile)

with open("data/repos.pkl", "wb") as outfile:
    pickle.dump(repos, outfile)

# Build a set of all unique packages across repositories
repo_names = list(repos.keys())
num_repos = len(repo_names)
all_packages = set(pkg for pkgs in repos.values() for pkg in pkgs)
all_packages = sorted(all_packages)  # sort for consistent ordering
num_packages = len(all_packages)

# Echo some details about the scraped data
print("Scraped",num_repos, "repositories and found",num_packages,"unique packages.")

# Create a repository-by-package binary matrix
X = np.zeros((num_repos, num_packages), dtype=int)

# Create a mapping from package names to indices
package_to_index = {pkg: idx for idx, pkg in enumerate(all_packages)}

for i, repo in enumerate(repo_names):
    for pkg in repos[repo]:
        j = package_to_index.get(pkg)
        if j is not None:
            X[i, j] = 1

# Use elbow method to determine the optimal number of clusters
elbow_method(X, fname="outputs/elbow.png")

# Set the optimal number of clusters based on the elbow method
optimal_k = 5

# Run kmeans
kmeans = KMeans(n_clusters=optimal_k, random_state=8675309)
clusters = kmeans.fit_predict(X)

# Run PCA to reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:, 1], X_pca[:, 0], c=clusters, cmap='viridis', alpha=0.7)

# List top 10 packages based on frequency

# Group the repository names by their cluster labels:
cluster_repo_map = {}
for repo, label in zip(repo_names, clusters):
    cluster_repo_map.setdefault(label, []).append(repo)

# Create a dictionary to store the top 10 packages for each cluster:
cluster_top_packages = {}
for label, repo_list in cluster_repo_map.items():
    # Collect all packages for repositories in this cluster:
    cluster_packages = []
    for repo in repo_list:
        # Look up dependencies for this repo in the 'repos' dictionary.
        cluster_packages.extend(repos[repo])
    counter = Counter(cluster_packages) # Count frequency of each package
    top_10 = counter.most_common(10)
    cluster_top_packages[label] = top_10 # Save result to dictionary

# Plot the top packages in each cluster
cluster_plot(optimal_k, clusters, X_pca, fname="outputs/clusters.png")

# Visualize the clusters
cluster_bar_charts(cluster_top_packages, optimal_k, fname="outputs/bar")