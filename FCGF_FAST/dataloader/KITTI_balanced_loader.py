from dataloader.generic_balanced_loader import GenericBalancedLoader
from dataloader.balanced.KITTI import KITTI_balanced

class KITTIBalancedPairDataset(GenericBalancedLoader):
    def __init__(self, phase, random_rotation):
        self.random_rotation = random_rotation
        self.phase = phase
        self.U = KITTI_balanced(phase)
        self.name = "KITTI"

