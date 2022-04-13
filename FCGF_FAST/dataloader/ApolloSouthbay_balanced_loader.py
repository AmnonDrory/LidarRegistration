from dataloader.generic_balanced_loader import GenericBalancedLoader
from dataloader.generic_refinement_loader import GenericRefinementLoader
from dataloader.balanced.ApolloSouthbay import ApolloSouthbay_balanced

class ApolloSouthbayBalancedPairDataset(GenericBalancedLoader):
    def __init__(self, phase, random_rotation):
        self.random_rotation = random_rotation
        self.phase = phase
        self.U = ApolloSouthbay_balanced(phase)
        self.name = "ApolloSouthbay"

class ApolloSouthbayRefinementDataset(GenericRefinementLoader):
    def __init__(self, *args, **kwargs):
        self.U = ApolloSouthbay_balanced('test')
        self.name = "ApolloSouthbay"

