"""Temperature scaling of class probabilities, shared by training and serving."""

import numpy as np

# Log of a zero probability would be minus infinity; this floor keeps scaling finite.
PROBABILITY_FLOOR = 1e-12


def scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Class probabilities after dividing their log scores by the temperature."""
    logits = np.log(np.clip(probabilities, PROBABILITY_FLOOR, 1.0)) / temperature
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)
