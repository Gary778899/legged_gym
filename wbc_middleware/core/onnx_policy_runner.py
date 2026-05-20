import numpy as np
import onnxruntime as ort


class OnnxPolicyRunner:
    def __init__(self, model_path: str, hidden_dim: int = 64, num_envs: int = 1):
        self.session = ort.InferenceSession(model_path)
        self.hidden_dim = hidden_dim
        self.num_envs = num_envs
        self.h_state = np.zeros((1, self.num_envs, self.hidden_dim), dtype=np.float32)
        self.c_state = np.zeros((1, self.num_envs, self.hidden_dim), dtype=np.float32)

    def run(self, obs: np.ndarray) -> np.ndarray:
        inputs = {
            "obs": obs,
            "hidden_state": self.h_state,
            "cell_state": self.c_state,
        }
        action, next_hidden_state, next_cell_state = self.session.run(None, inputs)
        self.h_state = next_hidden_state
        self.c_state = next_cell_state
        return action.flatten()

    def reset_memory(self) -> None:
        self.h_state.fill(0.0)
        self.c_state.fill(0.0)
