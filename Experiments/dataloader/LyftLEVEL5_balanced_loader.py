from dataloader.generic_balanced_loader import GenericBalancedLoader
from dataloader.balanced.LyftLEVEL5 import LyftLEVEL5_balanced

class LyftLEVEL5BalancedPairDataset(GenericBalancedLoader):
    def __init__(   self, 
                    phase,
                    transform=None,
                    random_rotation=True,
                    random_scale=True,
                    manual_seed=False,
                    config=None,
                    rank=None):
        
        self.U = LyftLEVEL5_balanced(phase.replace('val','validation'))
        GenericBalancedLoader.__init__(self, phase, transform, random_rotation, random_scale,
                              manual_seed, config, rank)
