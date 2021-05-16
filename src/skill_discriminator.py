from abc import ABC, abstractmethod


class SkillDiscriminator(ABC):

    @abstractmethod
    def train(self, batch):
        # assert that the input and output dimensions match the expected
        pass

    @abstractmethod
    def call(self, batch):
        # assert that the input and output dimensions match the expected
        pass