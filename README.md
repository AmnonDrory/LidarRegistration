# Stress Testing LiDAR Registration
To train and test point-cloud registration algorithms with automotive LiDAR datasets, one must select a subset of point-cloud pairs. Previously, simple heuristics were used to create such registration sets, e.g., using a spacing of 10 meters. We provide a smart algorithm that selects challenging sets, that include a balanced sampling from the various situations that appear in a dataset (various offsets, rotations, etc.). 

We provide registration sets that were produced by our algorithm, for the Apollo-Southbay and NuScenes datasets. In our paper, we use these sets to train and benchmark some recent and popular registration algorithms. We use FCGF deep features with a variety of robust motion-estimation algorithms. Surprisingly, we find that the fastest and most accurate results come not from recent algorithms such as Teaser++ and PointDSC, but rather from a modern version of RANSAC. 

Our paper: Stress Testing LiDAR Registration / Amnon Drory, Raja Giryes, Shai Avidan

## Contents of Directories

1. *BalancedDatasetGenerator*: Code for creating registration sets
2. *balanced_sets*: Contains the balanced sets used in the paper (as lists of frame indices and relative motions).

3. *Experiments*: Code used to run most of the experiments shown in the paper. Also used to train the PointDSC network. based on [PointDSC code](https://github.com/XuyangBai/PointDSC)
4. *FCGF_FAST*: Our code for training an FCGF network. Also includes code for the refinement experiment.
5. *DGR*: Somewhat modified code for training a DGR network. Based on [DGR code](https://github.com/chrischoy/DeepGlobalRegistration)
6. *GC-RANSAC*: Some additions to [GC-RANSAC code](https://github.com/danini/graph-cut-ransac), including Edge-Length Constraints. 

## Requirements
There are minimal requirements for the code to create registration sets. Reproducing our experiments, however, requires a variety of code packages, leading unfortunately to incompatibilities and
complications. In the *Requirements* directory we provided .yml files that detail two environments we used: 
one for experiments involving the GC-RANSAC code base, and one for other experiments. 
To recreate these environment, we recommend running:
```bash
conda env create -f requirements/basis.yml
```
And then installing the [minkowski engine](https://github.com/NVIDIA/MinkowskiEngine), which is needed
for FCGF and DGR. 
Additional code packages, such as GC-RANSAC and TEASER++, might not be able to coexist and will need to
be installed in separate environments. 

## Datasets 
Our balanced registration sets are selected from Apollo-Southbay and NuScenes. 
To run our code, one must have these datasets available locally. They can be downloaded from:
1. [Apollo-Southbay](https://apollo.auto/southbay.html)
2. [NuScenes](https://www.nuscenes.org/download) (Full dataset, v1.0, Trainval and Test)

The following files in our code need to be adjusted to reflect the paths to these datasets:
```
Experiments/dataloader/paths.py
BalancedDatasetGenerator/datasets/paths.py
FCGF_FR/dataloader/paths.py
DGR/dataloader/paths.py
```

## Example Command Lines
```bash
cd Experiments 
python -m test --fcgf_weights_file ../weights/ApolloSouthbay.400.pth --dataset A --algo RANSAC --mode GPF --iters 50000 # Testing RANSAC+GPF on Apollo-Southbay-Balanced
python -m test --fcgf_weights_file ../weights/ApolloSouthbay.400.pth --dataset B --algo RANSAC --mode MMN --iters 1000000 --GC_conf 0.9995 # Testing RANSAC with mutual-nearest neighbors filtering (MNN) cross-domain, on NuScenes-Boston-Balanced
python -m test --fcgf_weights_file ../weights/ApolloSouthbay.400.pth --dataset A --algo TEASER # Testing with TEASER++
python -m train --fcgf_weights_file ../weights/ApolloSouthbay.400.pth --dataset A # training PointDSC network on Apollo-Southbay-Balanced
python -m test --chosen_snapshot PointDSC_ApolloSouthbay_??????? --fcgf_weights_file ../weights/ApolloSouthbay.400.pth --dataset S --algo PointDSC # Testing PointDSC cross-domain, on NuScenes-Singapore-Balanced (replace question marks with actual output from previous command)
cd ..

cd FCGF_FAST
python -m train B # training FCGF network for dataset NuScenes-Boston-Balanced
./refinement_parallel.sh B # perform refinement test
cd ..

cd BalancedDatasetGenerator
python -m GenerateBalancedSet.py
```
