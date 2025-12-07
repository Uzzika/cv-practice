from abc import ABC, abstractmethod

class BaseDetector(ABC):
    def __init__(self, model_path, input_size):
        self.model_path = model_path
        self.input_size = input_size
        self.net = None

    @abstractmethod
    def preprocess(self, img):
        pass

    def forward(self, blob):
        self.net.setInput(blob)
        return self.net.forward()

    @abstractmethod
    def postprocess(self, outputs, img):
        pass

    def detect(self, img):
        blob = self.preprocess(img)
        outputs = self.forward(blob)
        return self.postprocess(outputs, img)