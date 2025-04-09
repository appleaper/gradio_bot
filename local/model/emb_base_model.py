from abc import ABC, abstractmethod


class EmbBase(ABC):
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def parse_single_sentence(self, sentence, model_name):
        pass

    @abstractmethod
    def unload_model(self):
        pass
