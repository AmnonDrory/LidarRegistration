## Modified DGR Code

DGR code is based on https://github.com/chrischoy/DeepGlobalRegistration. This version contains
substantial changes made as part of the research presented in the paper "Stress Testing LiDAR Registration" 
by Amnon Drory, Shai Avidan, and Raja Giryes, from Tel-Aviv University. 
These changes include:
* Adaptive failsafe threshold: we adapt the threshold so that the failsafe mechanism is applied to a reasonable ratio of inputs. The alternative, i.e. using a constant threshold, causes the failsafe to be applied to almost all inputs in some test sets.
* parallel training on multiple GPUs
* Testing in parallel. As of time of writing, the only way to make this work was using bash (some essential packages were incompatible with python and pytorch parallelism).
* Testing with FCGF features + RANSAC, optionally with mututal nearest neighbors filtering
* Support for new LiDAR registrations datasets, instead of KITTI-10m. These include e.g. NuScenes-Boston-Balanced and Apollo-Southbay-Balanced.
* Keeping track of statistics both before and after applying ICP refinement. 
* Augmentation with almost-planar random rotations, instead of with full 3D rotations (which are unnatural for LiDAR data).
* support for newer version of openCV (0.13), which provides a smarter and faster version of RANSAC
* support for newer version of Minkowski Engine (0.5)

