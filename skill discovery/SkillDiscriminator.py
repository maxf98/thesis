from abc import ABC, abstractmethod

class Discriminator(ABC):
    @abstractmethod
    def __init__(self,
                 num_skills,
                 ):
        pass

    @abstractmethod
    def train(self, batch):
        pass
