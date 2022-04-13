from time import time
try:
    import pygcransac
except Exception as E:
    print("Ignoring exception: " + str(E))
import numpy as np

def GC_RANSAC(A,B, distance_threshold, num_iterations, args, match_quality):
    
    x1y1z1_= np.ascontiguousarray(A)
    x2y2z2_= np.ascontiguousarray(B)
    params = { 
    'threshold': distance_threshold, 
    'conf': 0.999, 
    'spatial_coherence_weight': 0.0, 
    'max_iters': num_iterations, 
    'use_sprt': True, 
    'min_inlier_ratio_for_sprt': 0.1, 
    'sampler': 0, # default: 1, 0=RANSAC, 1=PROSAC
    'neighborhood': 0, 
    'neighborhood_size': 20, 
    }

    params['spatial_coherence_weight'] = args.spatial_coherence_weight
    params['sampler'] = args.prosac
    params['conf'] = args.GC_conf

    if args.fast_rejection == "NONE":
		params['use_sprt'] = False
	else:
        params['use_sprt'] = True # actually means "perform fast rejection"
    
    if args.fast_rejection == "ELC":
        params['min_inlier_ratio_for_sprt'] = -1 # negative value signals that c++ code should use ELC instead of SPRT
        
    if not args.GC_LO:
        params['neighborhood'] = 1 # non-zero value signals that c++ code should _not_ perform local-optimization

    if args.prosac:
        # sort from best quality to worst
        ord = np.argsort(-match_quality)
        x1y1z1_ = x1y1z1_[ord,:]
        x2y2z2_ = x2y2z2_[ord,:]

    start_time = time()
    pose_T, mask = pygcransac.findRigidTransform(
        x1y1z1_,
        x2y2z2_,
        **params)

    if pose_T is None:
        pose_T = np.eye(4, dtype=np.float32)

    elapsed_time = time() - start_time
    return pose_T.T, elapsed_time
