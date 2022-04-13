from dataloader.generic_balanced_loader import GenericBalancedLoader
from dataloader.balanced.LyftLEVEL5 import LyftLEVEL5_balanced

class LyftLEVEL5BalancedPairDataset(GenericBalancedLoader):
    def __init__(self, phase, random_rotation):
        self.random_rotation = random_rotation
        self.phase = phase
        self.U = LyftLEVEL5_balanced(phase)
        self.name = "LyftLEVEL5"
