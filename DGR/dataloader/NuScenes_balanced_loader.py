from dataloader.generic_balanced_loader import GenericBalancedLoader
from dataloader.balanced.NuScenes import NuScenes_balanced

class NuScenesBalancedPairDataset(GenericBalancedLoader):
    LOCATION = None
    def __init__(self, 
                    phase,
                    transform=None,
                    random_rotation=True,
                    random_scale=True,
                    manual_seed=False,
                    config=None,
                    rank=None):
        
        self.U = NuScenes_balanced(self.LOCATION, phase.replace('val','validation'))
        GenericBalancedLoader.__init__(self, phase, transform, random_rotation, random_scale,
                              manual_seed, config, rank)

        
class NuScenesBostonDataset(NuScenesBalancedPairDataset):
    LOCATION = 'boston'

class NuScenesSingaporeDataset(NuScenesBalancedPairDataset):
    LOCATION = 'singapore'    