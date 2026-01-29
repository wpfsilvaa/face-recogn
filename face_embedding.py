
import numpy as np
from deepface import DeepFace


def get_embedding(frame):
    """
    Gera o embedding facial usando DeepFace.

    Usar o backend 'opencv' é bem mais leve que 'retinaface' em CPU,
    mantendo boa qualidade para este caso de uso.
    """
    reps = DeepFace.represent(
        img_path=frame,
        model_name="Facenet",
        detector_backend="opencv",
        enforce_detection=False,
    )
    return np.array(reps[0]["embedding"])
