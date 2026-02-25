# Get folder path from command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  folder_path <- "." # Default to current directory
  cat("Using current directory\n")
} else {
  folder_path <- args[1]
  cat("Using folder:", folder_path, "\n")
}

require(Seurat)
require(SeuratObject)
require(tidyverse)
require(patchwork)
require(cowplot)
require(RColorBrewer)
require(Biobase)
require(clusterSim)
require(fpc)
require(clusterSim)
require(aricode)

lisi <- function(
    X, cell_labels, no_nn = 10, nn_eps = 0) {
  lisi <- NULL
  N <- nrow(X)
  dknn <- RANN::nn2(X, k = no_nn, eps = nn_eps)
  for (cell in 1:N) {
    nn.idx_cell <- dknn$nn.idx[cell, 2:ncol(dknn$nn.idx)]
    index_neighbor <- cell_labels[nn.idx_cell]
    prop_cell_labels_nn <- as.numeric(table(index_neighbor) / length(index_neighbor))
    inv_simpson_index <- 1 / sum(prop_cell_labels_nn^2)
    lisi <- c(lisi, inv_simpson_index)
  }
  return(lisi)
}

mean_nn <- function(
    vector_nn_neighbor,
    M) {
  print(vector_nn_neighbor)
  temp_matrix <- M[vector_nn_neighbor, ]
  return(apply(temp_matrix, MARGIN = 2, FUN = mean))
}

compute_final_embedding <- function(
    Matrix_nn_index,
    Matrix_query) {
  embedding_matrix <- NULL
  for (i in 1:dim(Matrix_nn_index)[1]) {
    embedding_matrix <- rbind(embedding_matrix, mean_nn(vector_nn_neighbor = Matrix_nn_index[i, ], M = Matrix_query))
  }
  return(embedding_matrix)
}

DimPlot <- function(
    data,
    dims = c(1, 2),
    cells = NULL,
    cols = NULL,
    pt.size = NULL,
    reduction = NULL,
    group.by = NULL,
    split.by = NULL,
    shape.by = NULL,
    order = NULL,
    shuffle = FALSE,
    seed = 1,
    label = FALSE,
    label.size = 4,
    label.color = "black",
    label.box = FALSE,
    repel = FALSE,
    cells.highlight = NULL,
    cols.highlight = "#DE2D26",
    sizes.highlight = 1,
    na.value = "grey50",
    ncol = NULL,
    combine = TRUE,
    raster = NULL,
    raster.dpi = c(512, 512)) {
  if (length(x = dims) != 2) {
    stop("'dims' must be a two-length vector")
  }
  colnames(data) <- paste0("UMAP", dims)
  data <- as.data.frame(x = data)
  dims <- paste0("UMAP", dims)
  data <- cbind(data, group.by)
  orig.groups <- group.by
  group.by <- colnames(x = data)[3:ncol(x = data)]
  for (group in group.by) {
    if (!is.factor(x = data[, group])) {
      data[, group] <- factor(x = data[, group])
    }
  }
  if (!is.null(x = shape.by)) {
    data[, shape.by] <- object[[shape.by, drop = TRUE]]
  }
  if (!is.null(x = split.by)) {
    data[, split.by] <- object[[split.by, drop = TRUE]]
  }
  if (isTRUE(x = shuffle)) {
    set.seed(seed = seed)
    data <- data[sample(x = 1:nrow(x = data)), ]
  }
  plots <- lapply(
    X = group.by,
    FUN = function(x) {
      plot <- SingleDimPlot(
        data = data[, c(dims, x, split.by, shape.by)],
        dims = dims,
        col.by = x,
        cols = cols,
        pt.size = pt.size,
        shape.by = shape.by,
        order = order,
        label = FALSE,
        cells.highlight = cells.highlight,
        cols.highlight = cols.highlight,
        sizes.highlight = sizes.highlight,
        na.value = na.value,
        raster = raster,
        raster.dpi = raster.dpi
      )
      if (label) {
        plot <- LabelClusters(
          plot = plot,
          id = x,
          repel = repel,
          size = label.size,
          split.by = split.by,
          box = label.box,
          color = label.color
        )
      }
      if (!is.null(x = split.by)) {
        plot <- plot + FacetTheme() +
          facet_wrap(
            facets = vars(!!sym(x = split.by)),
            ncol = if (length(x = group.by) > 1 || is.null(x = ncol)) {
              length(x = unique(x = data[, split.by]))
            } else {
              ncol
            }
          )
      }
      plot <- if (is.null(x = orig.groups)) {
        plot + labs(title = NULL)
      } else {
        plot + labs(title = NULL)
      }
    }
  )
  if (!is.null(x = split.by)) {
    ncol <- 1
  }
  if (combine) {
    plots <- wrap_plots(plots, ncol = orig.groups %iff% ncol)
  }
  return(plots)
}

set.seed(10)

# Brew colors for the UMAP
n <- 100
qual_col_pals <- brewer.pal.info[brewer.pal.info$category == "qual", ]
col_vector <- unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))

# Plug in your denoised embeddings...before computing UMAP but after PCA step (In what we discussed, these are the 15 dim embeddings)
# Say these are stored in denoised_embeddings.csv
Sys.setenv(RETICULATE_PYTHON = "/opt/homebrew/anaconda3/envs/mixed_diffusion/bin/python")

library(reticulate)
torch <- import("torch")

path <- file.path(folder_path, "denoising_results.pt")

# Correct: weights_only = FALSE (R logical) → Python False
obj <- torch$load(path, map_location = "cpu", weights_only = FALSE)

# Convert torch tensor to numpy array first, then to R matrix
jt_embedding_gmm_denoised <- as.matrix(obj$x_denoised$numpy())
data_together_pca_denoised <- jt_embedding_gmm_denoised # Check the dimension here...sometime R does weird things

jt_embedding_gmm_true <- as.matrix(obj$x_true$numpy())
data_together_pca_true <- jt_embedding_gmm_true # Check the dimension here...sometime R does weird things

cat("Denoised data shape:", dim(data_together_pca_denoised), "\n")
cat("True data shape:", dim(data_together_pca_true), "\n")


# You need to name the rows and columns

# Extract true labels from denoising_results.pt
true_labels_tensor <- obj$x_denoised_labels
true_labels <- as.vector(true_labels_tensor$numpy())
data_config <- obj$data_config
label_encoder <- data_config$label_encoder

# Convert numeric labels to string labels using the label encoder
# Create reverse mapping from numeric to string
label_encoder_list <- py_to_r(label_encoder)
reverse_label_encoder <- setNames(names(label_encoder_list), unlist(label_encoder_list))

# Convert true labels to string names
cell_names_raw <- sapply(true_labels, function(x) reverse_label_encoder[as.character(x)], USE.NAMES = FALSE, simplify = TRUE)

# Ensure cell_names is a character vector (handle any NULL or list issues)
cell_names <- as.character(unlist(cell_names_raw))

# Debug information
cat("Extracted true labels from denoising_results.pt\n")
cat("Class of cell_names:", class(cell_names), "\n")
cat("Length of cell_names:", length(cell_names), "\n")

# Handle unique labels safely
unique_labels <- unique(cell_names)
cat("Class of unique_labels:", class(unique_labels), "\n")
if (is.list(unique_labels)) {
  unique_labels <- as.character(unlist(unique_labels))
}
cat("Unique labels:", paste(unique_labels, collapse = ", "), "\n")
cat("Label distribution:\n")
print(table(cell_names))

# Process denoised data
rownames(data_together_pca_denoised) <- paste("p", 1:1:dim(data_together_pca_denoised)[1])
colnames(data_together_pca_denoised) <- paste("RNA", 1:dim(data_together_pca_denoised)[2])

rna_denoised <- as.sparse(t(data_together_pca_denoised))
colnames(rna_denoised) <- paste("p", 1:1:dim(data_together_pca_denoised)[1])

data_together_denoised <- CreateSeuratObject(counts = rna_denoised)

DefaultAssay(data_together_denoised) <- "RNA"

VariableFeatures(data_together_denoised) <- rownames(data_together_denoised[["RNA"]])
data_together_denoised[["pca"]] <- CreateDimReducObject(embeddings = as.matrix(data_together_pca_denoised), key = "PCA_", assay = DefaultAssay(data_together_denoised))

# Process true data
rownames(data_together_pca_true) <- paste("p", 1:1:dim(data_together_pca_true)[1])
colnames(data_together_pca_true) <- paste("RNA", 1:dim(data_together_pca_true)[2])

rna_true <- as.sparse(t(data_together_pca_true))
colnames(rna_true) <- paste("p", 1:1:dim(data_together_pca_true)[1])

data_together_true <- CreateSeuratObject(counts = rna_true)

DefaultAssay(data_together_true) <- "RNA"

VariableFeatures(data_together_true) <- rownames(data_together_true[["RNA"]])
data_together_true[["pca"]] <- CreateDimReducObject(embeddings = as.matrix(data_together_pca_true), key = "PCA_", assay = DefaultAssay(data_together_true))
data_together_denoised <- RunUMAP(data_together_denoised, reduction = "pca", dims = 1:ncol(data_together_pca_denoised), verbose = FALSE)
data_together_true <- RunUMAP(data_together_true, reduction = "pca", dims = 1:ncol(data_together_pca_true), verbose = FALSE)

data_atlas_denoised <- Embeddings(object = data_together_denoised, reduction = "umap")
data_atlas_true <- Embeddings(object = data_together_true, reduction = "umap")

p3_denoised <- DimPlot(data_atlas_denoised, group.by = cell_names, pt.size = 0.04, cols = col_vector[11:40]) + ggtitle("Denoised Data") + theme(plot.title = element_text(hjust = 0.5))
p3_true <- DimPlot(data_atlas_true, group.by = cell_names, pt.size = 0.04, cols = col_vector[11:40]) + ggtitle("True Data") + theme(plot.title = element_text(hjust = 0.5))

# Quality metrics

# Computing Metrics
# Silhouette Score

names_unique <- unique(cell_names)
cell_names_integer <- rep(0, length(names_unique))
for (count in seq_along(names_unique))
{
  places_where_present <- (cell_names == names_unique[count])
  cell_names_integer[places_where_present] <- count
}

# Function to test different resolutions and find optimal ARI
test_resolutions <- function(data_seurat, data_name) {
  cat("\n=== Testing different resolutions for", data_name, "===\n")

  # Test range of resolutions
  resolutions <- c(seq(0.01, 0.2, by = 0.01), seq(0.2, 2.0, by = 0.1))
  ari_results <- data.frame(
    Resolution = resolutions, ARI = numeric(length(resolutions)),
    N_Clusters = numeric(length(resolutions))
  )

  # Find neighbors once (this is expensive)
  data_seurat <- FindNeighbors(data_seurat, reduction = "umap", dims = 1:2, verbose = FALSE)

  for (i in seq_along(resolutions)) {
    res <- resolutions[i]

    # Find clusters with current resolution
    data_seurat <- FindClusters(data_seurat, resolution = res, method = 4, verbose = FALSE)
    seurat_cluster <- data_seurat$seurat_clusters

    # Compute ARI
    ari <- mclust::adjustedRandIndex(seurat_cluster, cell_names_integer)
    n_clusters <- length(unique(seurat_cluster))

    ari_results$ARI[i] <- ari
    ari_results$N_Clusters[i] <- n_clusters

    cat(sprintf("Resolution: %.3f | ARI: %.4f | N_Clusters: %d\n", res, ari, n_clusters))
  }

  # Find best resolution
  best_idx <- which.max(ari_results$ARI)
  best_resolution <- ari_results$Resolution[best_idx]
  best_ari <- ari_results$ARI[best_idx]

  cat(sprintf("\n*** BEST RESOLUTION: %.2f (ARI: %.4f) ***\n", best_resolution, best_ari))

  # Safe handling of unique cell_names_integer
  unique_cell_types <- unique(cell_names_integer)
  if (is.list(unique_cell_types)) {
    unique_cell_types <- as.numeric(unlist(unique_cell_types))
  }
  cat("Number of true cell types:", length(unique_cell_types), "\n")

  return(list(results = ari_results, best_resolution = best_resolution, best_ari = best_ari))
}

# Function to compute silhouette score using true labels as benchmark
compute_silhouette_benchmark <- function(data_atlas, true_labels, data_name) {
  cat("\n=== Computing silhouette benchmark for", data_name, "===\n")

  # Convert true labels to numeric if they aren't already
  if (!is.numeric(true_labels)) {
    true_labels <- as.numeric(as.factor(true_labels))
  }

  # Calculate silhouette score using true labels
  silhouette_result <- cluster::silhouette(true_labels, dist(data_atlas))
  silhouette_benchmark <- mean(silhouette_result[, 3])

  cat("Silhouette score with true labels:", round(silhouette_benchmark, 4), "\n")

  return(silhouette_benchmark)
}

# Function to compute all metrics for a dataset
compute_metrics <- function(data_seurat, data_atlas, data_name, resolution = NULL) {
  cat("\n=== Computing metrics for", data_name, "===\n")

  # Average Rand Index
  data_seurat <- FindNeighbors(data_seurat, reduction = "umap", dims = 1:2)

  # Use provided resolution or default
  if (is.null(resolution)) {
    resolution <- 0.8
  }
  cat("Using resolution:", resolution, "\n")

  data_seurat <- FindClusters(data_seurat, resolution = resolution, method = 4)
  seurat_cluster <- data_seurat$seurat_clusters
  adj_r_index <- mclust::adjustedRandIndex(seurat_cluster, cell_names_integer)

  # V Measure
  v_measure <- clevr::v_measure(seurat_cluster, cell_names_integer)

  ## LISI Score
  lisi_score <- mean(lisi(data_atlas, cell_names_integer))

  ## Silhouette Score
  silhouette_score <- mean(cluster::silhouette(
    as.numeric(seurat_cluster),
    dist(data_atlas)
  )[, 3])

  ## Normalized Mutual Information (NMI)
  # Use aricode::NMI, ensure inputs are vectors/factors
  nmi <- tryCatch(
    {
      aricode::NMI(as.vector(seurat_cluster), as.vector(cell_names_integer))
    },
    error = function(e) {
      warning(paste("Failed to compute NMI:", e))
      NA
    }
  )




  return(list(
    adj_r_index = adj_r_index,
    nmi = nmi,
    v_measure = v_measure,
    lisi_score = lisi_score,
    silhouette_score = silhouette_score,
    seurat_cluster = seurat_cluster
  ))
}

# Test different resolutions to find optimal ARI
resolution_test_denoised <- test_resolutions(data_together_denoised, "Denoised Data")
resolution_test_true <- test_resolutions(data_together_true, "True Data")

# Save resolution test results
write.csv(resolution_test_denoised$results, file.path(folder_path, "resolution_test_denoised.csv"), row.names = FALSE)
write.csv(resolution_test_true$results, file.path(folder_path, "resolution_test_true.csv"), row.names = FALSE)

# Compute silhouette benchmarks using true labels
silhouette_benchmark_denoised <- compute_silhouette_benchmark(data_atlas_denoised, cell_names_integer, "Denoised Data")
silhouette_benchmark_true <- compute_silhouette_benchmark(data_atlas_true, cell_names_integer, "True Data")

# Compute metrics for both datasets using optimal resolutions
metrics_denoised <- compute_metrics(data_together_denoised, data_atlas_denoised, "Denoised Data",
  resolution = resolution_test_denoised$best_resolution
)
metrics_true <- compute_metrics(data_together_true, data_atlas_true, "True Data",
  resolution = resolution_test_true$best_resolution
)

# Print results summary
cat("\n=== Clustering Quality Metrics Comparison ===\n")
cat("Dataset: Denoised\n")
cat("  Adjusted Rand Index:", metrics_denoised$adj_r_index, "\n")
cat("  V-Measure:", metrics_denoised$v_measure, "\n")
cat("  Mean LISI Score:", metrics_denoised$lisi_score, "\n")
cat("  Mean Silhouette Score:", metrics_denoised$silhouette_score, "\n")
cat("  Normalized Mutual Information:", metrics_denoised$nmi, "\n")
cat("  Silhouette Benchmark (True Labels):", silhouette_benchmark_denoised, "\n")

cat("\nDataset: True\n")
cat("  Adjusted Rand Index:", metrics_true$adj_r_index, "\n")
cat("  V-Measure:", metrics_true$v_measure, "\n")
cat("  Mean LISI Score:", metrics_true$lisi_score, "\n")
cat("  Mean Silhouette Score:", metrics_true$silhouette_score, "\n")
cat("  Normalized Mutual Information:", metrics_true$nmi, "\n")
cat("  Silhouette Benchmark (True Labels):", silhouette_benchmark_true, "\n")

# Save metrics to CSV
results <- data.frame(
  Dataset = rep(c("Denoised", "True"), each = 6),
  Metric = rep(c("Adjusted_Rand_Index", "V_Measure", "Normalized_Mutual_Information", "Mean_LISI_Score", "Mean_Silhouette_Score", "Silhouette_Benchmark"), 2),
  Value = c(
    metrics_denoised$adj_r_index, metrics_denoised$v_measure, metrics_denoised$nmi,
    metrics_denoised$lisi_score, metrics_denoised$silhouette_score, silhouette_benchmark_denoised,
    metrics_true$adj_r_index, metrics_true$v_measure, metrics_true$nmi,
    metrics_true$lisi_score, metrics_true$silhouette_score, silhouette_benchmark_true
  )
)

write.csv(results, file.path(folder_path, "clustering_metrics_results.csv"), row.names = FALSE)
cat("\nResults saved to:", file.path(folder_path, "clustering_metrics_results.csv"), "\n")

# Create UMAP plots
library(ggplot2)

# Get UMAP coordinates for denoised data
umap_coords_denoised <- data.frame(
  UMAP1 = data_atlas_denoised[, 1],
  UMAP2 = data_atlas_denoised[, 2],
  true_labels = as.factor(cell_names_integer),
  cluster_labels = as.factor(metrics_denoised$seurat_cluster)
)

# Get UMAP coordinates for true data
umap_coords_true <- data.frame(
  UMAP1 = data_atlas_true[, 1],
  UMAP2 = data_atlas_true[, 2],
  true_labels = as.factor(cell_names_integer),
  cluster_labels = as.factor(metrics_true$seurat_cluster)
)

# Plot with true labels - Denoised
p1_denoised <- ggplot(umap_coords_denoised, aes(x = UMAP1, y = UMAP2, color = true_labels)) +
  geom_point(size = 2, alpha = 0.7) +
  theme_minimal() +
  labs(
    title = "UMAP: Denoised Data - True Cell Labels",
    x = "UMAP1", y = "UMAP2", color = "Cell Type"
  ) +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

# Plot with K-means clusters - Denoised
p2_denoised <- ggplot(umap_coords_denoised, aes(x = UMAP1, y = UMAP2, color = cluster_labels)) +
  geom_point(size = 2, alpha = 0.7) +
  theme_minimal() +
  labs(
    title = "UMAP: Denoised Data - K-means Clusters",
    x = "UMAP1", y = "UMAP2", color = "Cluster"
  ) +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

# Plot with true labels - True
p1_true <- ggplot(umap_coords_true, aes(x = UMAP1, y = UMAP2, color = true_labels)) +
  geom_point(size = 2, alpha = 0.7) +
  theme_minimal() +
  labs(
    title = "UMAP: True Data - True Cell Labels",
    x = "UMAP1", y = "UMAP2", color = "Cell Type"
  ) +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

# Plot with K-means clusters - True
p2_true <- ggplot(umap_coords_true, aes(x = UMAP1, y = UMAP2, color = cluster_labels)) +
  geom_point(size = 2, alpha = 0.7) +
  theme_minimal() +
  labs(
    title = "UMAP: True Data - K-means Clusters",
    x = "UMAP1", y = "UMAP2", color = "Cluster"
  ) +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

# Save plots as a combined subplot
combined_plot <- (p1_denoised + p1_true) / (p2_denoised + p2_true)
ggsave(file.path(folder_path, "clustering_analysis_comparison.png"), combined_plot, width = 20, height = 12, dpi = 300)

cat("Combined subplot saved to:", file.path(folder_path, "clustering_analysis_comparison.png"), "\n")
