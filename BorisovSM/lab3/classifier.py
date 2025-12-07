from abc import ABC, abstractmethod


class Classifier(ABC):
    def __init__(self, args):
        self.args = args

    @staticmethod
    def create(args):
        if args.algo == "bow":
            from bow import BoWClassifier
            return BoWClassifier(args)
        elif args.algo == "cnn":
            from cnn import CNNClassifier
            return CNNClassifier(args)
        else:
            raise ValueError(f"Неподдерживаемый алгоритм: {args.algo}")

    @abstractmethod
    def train(self, train_items):
        ...

    @abstractmethod
    def test(self, test_items):
        ...
