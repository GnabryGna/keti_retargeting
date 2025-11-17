import numpy as np
import torch


def to_numpy(data):
    """
    Recursively converts data (dict, list, torch.Tensor, scalar) to numpy arrays.

    Args:
        data: Input data to convert.
            - dict: Recursively converts each value.
            - list or tuple: Converts each element.
            - torch.Tensor: Converts to numpy.
            - scalar (int, float): Converts to numpy scalar.

    Returns:
        Converted data in numpy format.
    """
    if isinstance(data, dict):
        return {key: to_numpy(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        converted = [to_numpy(item) for item in data]
        return np.array(converted) if isinstance(data, list) else tuple(converted)
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, (int, float)):
        return np.array(data)
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def to_torch(data, device="cpu"):
    """
    Recursively converts data (dict, list, np.ndarray, scalar) to torch tensors.

    Args:
        data: Input data to convert.
            - dict: Recursively converts each value.
            - list or tuple: Converts each element.
            - np.ndarray: Converts to torch.Tensor.
            - scalar (int, float): Converts to torch scalar.
        device: The device to place the tensors on (e.g., "cpu" or "cuda").

    Returns:
        Converted data in torch.Tensor format.
    """
    if isinstance(data, dict):
        return {key: to_torch(value, device=device) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        converted = [to_torch(item, device=device) for item in data]
        return torch.stack(converted)
    elif isinstance(data, np.ndarray):
        return torch.tensor(data, device=device, dtype=torch.float32)
    elif isinstance(data, (int, float)):
        return torch.tensor(data, device=device, dtype=torch.float32)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
