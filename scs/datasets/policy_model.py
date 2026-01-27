from __future__ import annotations

from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
)

import d3rlpy
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.torch.policies import NormalPolicy
import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    import gymnasium as gym


def load_model_data_from_d4rl(
    dataset: str,
) -> tuple[dict[str, dict[str, np.ndarray]], gym.Env]:
    try:
        _, env = d3rlpy.datasets.get_d4rl(dataset)
        # Since we want the d4rtl metadata
        data = env.get_dataset()  # type: ignore[attr-defined]
        model_data: dict[str, dict[str, np.ndarray]] = {
            "metadata": {
                "input_dimension": np.array(data["observations"].shape[1]),
                "output_dimension": np.array(data["actions"].shape[1]),
                "nonlinearity": data["metadata/policy/nonlinearity"],
                "output_distribution": data["metadata/policy/output_distribution"],
            },
        }
        for key in data.keys():
            elements = key.split("/")
            if len(elements) < 4 or not elements[1] == "policy":
                continue
            if elements[2] not in model_data:
                model_data[elements[2]] = {elements[3]: data[key]}
            else:
                model_data[elements[2]][elements[3]] = data[key]
    except Exception as e:
        print(f"Failed to load model data from d4rl dataset '{dataset}'")
        raise e
    return model_data, env  # type: ignore[assignment]


def numpy_io(func: Callable) -> Callable:
    """Decorator to handle numpy array input/output conversion and batch dimension
    handling for the 'TanhGaussianFromNormal' class.
    """

    @wraps(func)
    def wrapper(self, x: np.ndarray) -> Any:
        # get device and convert input
        device = next(self.parameters()).device
        xt = torch.as_tensor(x, dtype=torch.float32, device=device)

        # add batch dimension if missing
        added_batch_dim = False
        if xt.dim() == 1:
            xt = xt.unsqueeze(0)
            added_batch_dim = True

        result = func(self, xt)

        # remove batch dimension if it was added
        if isinstance(result, tuple):
            outputs = []
            for tensor in result:
                if added_batch_dim and tensor.dim() > 0:
                    tensor = tensor.squeeze(0)
                outputs.append(tensor.detach().cpu().numpy())
            return tuple(outputs)
        else:
            if added_batch_dim and result.dim() > 0:
                result = result.squeeze(0)
            return result.detach().cpu().numpy()

    return wrapper


class TanhGaussianFromNormal(NormalPolicy):
    """
    Extends d3rlpy's NormalPolicy to:
      - Sample u = mu + std * eps
      - Squash with tanh(u)
      - Return (action, mean, std)
    """

    @numpy_io
    def get_policy(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and std of the (unsquashed) Gaussian policy."""
        mu, _, logstd = self.forward(x)
        std = torch.exp(logstd)  # type: ignore[reportArgumentType]
        return mu, std

    @numpy_io
    def action(self, x: torch.Tensor) -> torch.Tensor:
        """Sample an action from the tanh-squashed Gaussian policy."""
        action, _, _ = self.sample(x)
        return action

    @numpy_io
    def sample_np_io(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sample(x)

    def sample(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # get mean and clamped log-std from NormalPolicy.forward
        mu, _, logstd = self.forward(x)
        std = torch.exp(logstd)  # type: ignore[reportArgumentType]
        # sample from the Gaussian
        sample = torch.randn_like(mu)
        a = mu + std * sample
        action = torch.tanh(a)
        return action, mu, std

    @classmethod
    def create(
        cls,
        model_data: dict[str, dict[str, np.ndarray]],
        min_logstd: float = -20.0,
        max_logstd: float = 2.0,
        use_gpu: bool = False,
    ) -> TanhGaussianFromNormal:
        device = (
            torch.device("cuda")
            if (use_gpu and torch.cuda.is_available())
            else torch.device("cpu")
        )
        md = model_data["metadata"]
        activation = cls._decode_activation(md["nonlinearity"])
        hidden_units = [model_data[f]["weight"].shape[0] for f in ("fc0", "fc1")]

        encoder = VectorEncoderFactory(
            hidden_units=hidden_units,
            activation=activation,
            use_batch_norm=False,
        ).create(observation_shape=(int(md["input_dimension"]),))

        policy = cls(
            encoder=encoder,
            hidden_size=hidden_units[-1],
            action_size=int(md["output_dimension"]),
            min_logstd=min_logstd,
            max_logstd=max_logstd,
            use_std_parameter=False,
        ).to(device)

        state_dict = cls._build_state_dict(model_data)
        policy.load_state_dict(state_dict, strict=True)

        return policy

    @classmethod
    def create_from_name(
        cls,
        dataset_name: str,
        min_logstd: float = -20.0,
        max_logstd: float = 2.0,
        use_gpu: bool = False,
    ) -> TanhGaussianFromNormal:
        model_data, _ = load_model_data_from_d4rl(dataset_name)
        return cls.create(
            model_data,
            min_logstd=min_logstd,
            max_logstd=max_logstd,
            use_gpu=use_gpu,
        )

    @staticmethod
    def _decode_activation(act) -> str:
        if isinstance(act, (bytes, bytearray)):
            return act.decode("utf-8")
        return str(act)

    @staticmethod
    def _build_state_dict(
        model_data: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, torch.Tensor]:
        sd = {}
        # encoder linears
        for i, name in enumerate(("fc0", "fc1")):
            layer_idx = i * 2  # d3rlpy uses 0, 2 for layer indices (with ReLU at 1, 3)
            prefix = f"_encoder._layers.{layer_idx}"
            sd[f"{prefix}.weight"] = torch.from_numpy(model_data[name]["weight"])
            sd[f"{prefix}.bias"] = torch.from_numpy(model_data[name]["bias"])
        # mu & logstd heads
        sd["_mu.weight"] = torch.from_numpy(model_data["last_fc"]["weight"])
        sd["_mu.bias"] = torch.from_numpy(model_data["last_fc"]["bias"])
        sd["_logstd.weight"] = torch.from_numpy(model_data["last_fc_log_std"]["weight"])
        sd["_logstd.bias"] = torch.from_numpy(model_data["last_fc_log_std"]["bias"])
        return sd
