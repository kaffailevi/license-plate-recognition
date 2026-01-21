import cv2
import torch

from src.camera import Camera
from src.model_manager import ModelManager
from src.inference import InferenceEngine

MODEL_DIR = "./models"
NUM_CLASSES = 2        # background + license plate
FRAME_SKIP = 2

def main():
    torch.set_num_threads(4)

    camera = Camera()
    model_manager = ModelManager(MODEL_DIR, NUM_CLASSES)
    inference_engine = InferenceEngine(model_manager.get_models())

    frame_id = 0

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        if frame_id % FRAME_SKIP == 0:
            results = inference_engine.run(frame)

            for model_name, output in results:
                print(
                    model_name,
                    output["boxes"].shape,
                    output["scores"].mean().item() if len(output["scores"]) > 0 else 0
                )

        frame_id += 1

        cv2.imshow("Live Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
