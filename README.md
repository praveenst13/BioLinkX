# Heterogeneous Graph Neural Networks for Multi-Omics Biomedical Data Integration
## Overview
    This repository provides a Python implementation of a heterogeneous graph neural network (HGNN) pipeline for integrating multi-omics data and biomedical knowledge networks to predict disease-gene associations. The framework supports diverse node and edge types—representing diseases, genes, drugs, proteins, functions, and side effects—using the PyTorch Geometric ecosystem.Core biomedical entities are equipped with rich features from transcriptomics, epigenomics, and proteomics data through advanced feature engineering and dimensionality reduction (PCA). For nodes lacking direct experimental measurements, learnable embeddings are provided.
    The pipeline is optimized for scalable, high-throughput AI environments and offers production-grade reliability, with particular alignment for IBM Z systems to achieve secure, real-time, enterprise-grade inferencing.

## Problem Statement
    Biomedical and clinical domains generate massive multi-omics datasets spanning molecular (omics), network, and phenotypic layers. Existing computational methods struggle to: meaningfully integrate heterogeneous data,capture complex, multi-scale biological relationships,and deliver high performance link predictions (e.g., disease-gene associations) at real-world scale.Traditional omics integration methods often ignore cross-modal biological interactions, and many graph-based models do not scale or fail to provide interpretable results necessary in translational and clinical settings.

## Solution
    This project introduces a robust, modular pipeline that:

        1)Builds a heterogeneous biomedical knowledge graph integrating entities and relationships from standard public resources and omics files.

        2)Processes and integrates multiple omics layers (transcriptomics, epigenomics, proteomics) into fixed-length node features using normalization and dimensionality reduction.

        3)Dynamically assigns features: Uses fixed omics features when available, and learnable embeddings when not.

        4)Leverages a 3-layer GraphSAGE-based encoder from PyTorch Geometric for deep neighborhood learning, followed by a neural link predictor for classification tasks.
        
        5)Implements a full training, validation, and test loop for link prediction (with robust handling of device placement, incomplete/missing data, and persistent checkpointing).

## Key Features
      *)Flexible Data Integration: Ready-to-use loaders for omics and network datasets.

      *)Scalable Model Architecture: GraphSAGE encoder and heterogeneous convolutions; suitable for large graphs.

      *)Hybrid Feature Assignment: Handles missing data and diverse biological measurements robustly.

      *)Comprehensive Evaluation: Supports ROC-AUC and average precision metrics.

      *)Production Readiness: Device-agnostic, checkpoint-enabled, and robust to failures.

## Code Sample:
```py
class DeeperGraphSAGEEncoder(torch.nn.Module):
    """
    A deeper, memory-optimized GNN encoder using SAGEConv.
    Uses three layers for multi-hop neighborhood aggregation.
    """
    def __init__(self, metadata, hidden_channels, out_channels):
        super().__init__()

        # Helper to dynamically create conv layers for existing edge types
        def get_conv_dict(conv_layer, h_channels):
            conv_dict = {}
            for rel in metadata[1]:
                src, rel_type, dst = rel
                # Check if the edge type exists and has edges in the global 'data' object
                if (src, rel_type, dst) in data.edge_types and data[(src, rel_type, dst)].edge_index.numel() > 0:
                    # (-1, -1) for heterogeneous input features; SAGEConv uses mean aggregation by default
                    conv_dict[rel] = conv_layer((-1, -1), h_channels)
            return conv_dict

        # Three layers of SAGEConv for multi-hop neighborhood aggregation
        self.conv1 = HeteroConv(get_conv_dict(SAGEConv, hidden_channels), aggr='sum')
        self.conv2 = HeteroConv(get_conv_dict(SAGEConv, hidden_channels), aggr='sum')
        # Final layer outputs the desired dimension
        self.conv3 = HeteroConv(get_conv_dict(SAGEConv, out_channels), aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        # Layer 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        # Layer 2
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        # Layer 3 (Output layer)
        x_dict = self.conv3(x_dict, edge_index_dict)
        return x_dict

class LinkPredictor(torch.nn.Module):
    """MLP classifier for link prediction using concatenated embeddings."""
    def __init__(self, in_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * in_channels, in_channels)
        self.lin2 = torch.nn.Linear(in_channels, in_channels // 2)
        self.lin3 = torch.nn.Linear(in_channels // 2, 1)
        self.dropout = torch.nn.Dropout(0.3)
        self.bn = torch.nn.BatchNorm1d(in_channels)

    def forward(self, z_src, z_dst, edge_label_index):
        src_index, dst_index = edge_label_index
        # Concatenate source and destination embeddings
        z = torch.cat([z_src[src_index], z_dst[dst_index]], dim=-1)
        z = self.lin1(z)
        # BatchNorm expects (batch_size, num_features)
        self.bn.num_features = z.shape[1]
        z = self.bn(z)
        z = z.relu()
        z = self.dropout(z)
        z = self.lin2(z).relu()
        return self.lin3(z).squeeze()
```

## Output

<img width="834" height="561" alt="image" src="https://github.com/user-attachments/assets/c2677d59-b875-4869-b2e8-19c760cd07d6" />

<img width="693" height="550" alt="image" src="https://github.com/user-attachments/assets/5b2ccc06-6cce-4240-88f8-08a7bea5f697" />

<img width="931" height="610" alt="image" src="https://github.com/user-attachments/assets/869a326b-3afd-4eba-b4d0-a0982c568112" />

## Conclusion
  Overall, this study demonstrates the power and potential of heterogeneous graph neural networks as a transformative tool for integrating diverse biomedical data and accelerating translational research in complex diseases
## License
  MIT License
