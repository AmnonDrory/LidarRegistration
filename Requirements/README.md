## Requirements
The requirements for the code to create registration sets are minimal. Reproducing our experiments, however, requires a variety of code packages, leading unfortunately to incompatibilities and
complications. The provided .yml files  detail two environments we used: 
one for experiments involving the GC-RANSAC code base, and one for other experiments. 
To recreate these environment, we recommend running:
```bash
conda env create -f requirements/basis.yml
```
And then installing the [minkowski engine](https://github.com/NVIDIA/MinkowskiEngine), which is needed
for FCGF and DGR. 
Additional code packages, such as GC-RANSAC and TEASER++, might not be able to coexist and will need to
be installed in separate environments. 
