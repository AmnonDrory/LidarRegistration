## Balanced Registration Set Generator

This directory contains code that creates a challenging and balanced set of LiDAR scan pairs, 
for training and testing point-cloud registration algorithms. 
The input is a LiDAR dataset (e.g. KITTI, NuScenens, Apollo-Southbay), 
which consists of sequences of point clouds recorded by a vehicle
during a driving session. From these sequences, this algorithms 
selects a subset of pairs which provides a diverse set of registration
challenges. This set contains pairs with a variety of time offsets, 
relative motions and source sequences. This is achieved by:
1. Selecting a set of candidate pairs from all source sequences, with 
   diverse time offsets, and sufficient overlap between the two point clouds.  
2. Repeatedly:
   a. sampling uniformy at random a 6DOF motion. 
   b. Selecting a candidate pair whose relative motion is close to
      the motion from (a). 
