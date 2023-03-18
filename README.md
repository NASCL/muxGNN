# Multiplex Graph Neural Network (muxGNN)

---
## Overview

---
## Setup

This implementation is based on DGL. We recommend that you install PyTorch and DGL separately according to your version of CUDA. The required version of DGL is 0.7.*

You can install the remaining dependencies by running:

```bash
pip install -r requirements.txt
```

---
## Datasets

### Link Prediction

The DGL graphs with node features and val/test edge sets for each dataset are provided in the [data](/data) directory.

- The Amazon and YouTube datasets along with their train/val/test splits and feature sets are from [Cen et al. (2019)](https://github.com/THUDM/GATNE).
- The Twitter dataset is sampled from [Source](https://snap.stanford.edu/data/higgs-twitter.html).
- The Tissue-PPI dataset is sampled from [Source](http://snap.stanford.edu/ohmnet/).

For the Twitter dataset, the 3-layer reply, retweet, and mention network is sampled for all nodes which participate in at least one reply. We then extract the largest strongly connected component in this network for our experiments.
10% of edges are randomly sampled for a test set and 5% of edges for a validation set. An equivalent number of negative edges is added to each test and validation set.

For the Tissue-PPI data, two datasets are generated for the transductive and inductive experiments. Both datasets are derived from the 10-layer network consisting of the ten largest tissue layers in the original data.
For the transductive experiments, each edge was randomly split into 5 cross-validation folds. For the inductive experiment, 15% of nodes were masked from the graph.
20% of remaining edges were sampled as a validation set and removed from the training graph. For evaluation, 50% of the edges incident on the removed nodes were added to the training graph,
and the remaining 50% of edges, along with an equivalent number of randomly sampled negative edges were used as a test set.

### Graph Classification

The graph classification datasets are taken from [StreamSpot](https://sbustreamspot.github.io/) and from [Unicorn](https://www.ndss-symposium.org/ndss-paper/unicorn-runtime-provenance-based-detector-for-advanced-persistent-threats/).
The processed DGL graphs are available on [IEEE DataPort](https://dx.doi.org/10.21227/92nh-5n87)

---

## Usage

Execute the following script to train and evaluate muxGNN on link prediction or graph classification.

```bash
python -um main_linkpred <dataset> <model-name>
```

Options for datasets are 'amazon', 'twitter', 'youtube', or 'tissue_ppi'.  
Set the ```--inductive``` flag with the 'tissue_ppi' dataset to run the inductive link prediction experiment.

```bash
python -um main_graphpred <dataset> <model-nam>
```

Options for datasets are 'streamspot' or 'wget'.

The model-name is a name of your choice for where the logs and saved models will be stored.

See each script's help for the full set of optional training arguments.

