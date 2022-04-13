from dataloader.generic_balanced_loader import GenericBalancedLoader
from dataloader.balanced.NuScenes import NuScenes_balanced

class NuScenesBalancedPairDataset(GenericBalancedLoader):
    LOCATION = None
    def __init__(self, phase, random_rotation):
        self.random_rotation = random_rotation
        self.phase = phase
        self.U = NuScenes_balanced(self.LOCATION, phase)

        
class NuScenesBostonDataset(NuScenesBalancedPairDataset):
    LOCATION = 'boston'
    name = 'NuScenesBoston'

class NuScenesSingaporeDataset(NuScenesBalancedPairDataset):
    LOCATION = 'singapore'    
    name = 'NuScenesSingapore'