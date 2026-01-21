import cv2
import torch
import numpy as np

class InferenceEngine:
    def __init__(self, models):
        self.models = models

    def preprocess(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img)

    def run(self, frame):
        input_tensor = self.preprocess(frame)

        results = []
        with torch.no_grad():
            for name, model in self.models:
                output = model([input_tensor])[0]
                results.append((name, output))

        return results
