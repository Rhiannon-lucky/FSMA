from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple
import abc
import numpy as np
from typing import List, Optional, Tuple, Any
# from preprocessor import Preprocessor
from typing import TYPE_CHECKING, Callable, List, Dict, Optional, Tuple, Union
CLIP_VALUES_TYPE = Tuple[Union[int, float, np.ndarray], Union[int, float, np.ndarray]]  # pylint: disable=C0103

# from art.utils import CLIP_VALUES_TYPE


logger = logging.getLogger(__name__)

class Preprocessor(abc.ABC):
    """
    Abstract base class for preprocessing defences.

    By default, the gradient is estimated using BPDA with the identity function.
        To modify, override `estimate_gradient`
    """

    params: List[str] = []

    def __init__(self, is_fitted: bool = False, apply_fit: bool = True, apply_predict: bool = True) -> None:
        """
        Create a preprocessing object.

        Optionally, set attributes.
        """
        self._is_fitted = bool(is_fitted)
        self._apply_fit = bool(apply_fit)
        self._apply_predict = bool(apply_predict)

    @property
    def is_fitted(self) -> bool:
        """
        Return the state of the preprocessing object.

        :return: `True` if the preprocessing model has been fitted (if this applies).
        """
        return self._is_fitted

    @property
    def apply_fit(self) -> bool:
        """
        Property of the defence indicating if it should be applied at training time.

        :return: `True` if the defence should be applied when fitting a model, `False` otherwise.
        """
        return self._apply_fit

    @property
    def apply_predict(self) -> bool:
        """
        Property of the defence indicating if it should be applied at test time.

        :return: `True` if the defence should be applied at prediction time, `False` otherwise.
        """
        return self._apply_predict

    @abc.abstractmethod
    def __call__(self, x: np.ndarray, y: Optional[Any] = None) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Perform data preprocessing and return preprocessed data as tuple.

        :param x: Dataset to be preprocessed.
        :param y: Labels to be preprocessed.
        :return: Preprocessed data.
        """
        raise NotImplementedError

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        Fit the parameters of the data preprocessor if it has any.

        :param x: Training set to fit the preprocessor.
        :param y: Labels for the training set.
        :param kwargs: Other parameters.
        """
        pass

    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:  # pylint: disable=W0613,R0201
        """
        Provide an estimate of the gradients of the defence for the backward pass. If the defence is not differentiable,
        this is an estimate of the gradient, most often replacing the computation performed by the defence with the
        identity function (the default).

        :param x: Input data for which the gradient is estimated. First dimension is the batch size.
        :param grad: Gradient value so far.
        :return: The gradient (estimate) of the defence.
        """
        return grad

    def set_params(self, **kwargs) -> None:  # pragma: no cover
        """
        Take in a dictionary of parameters and apply checks before saving them as attributes.
        """
        for key, value in kwargs.items():
            if key in self.params:
                setattr(self, key, value)
        self._check_params()

    def _check_params(self) -> None:  # pragma: no cover
        pass

    def forward(self, x: Any, y: Any = None) -> Tuple[Any, Any]:
        """
        Perform data preprocessing and return preprocessed data.

        :param x: Dataset to be preprocessed.
        :param y: Labels to be preprocessed.
        :return: Preprocessed data.
        """
        raise NotImplementedError


class PreprocessorPyTorch(Preprocessor):
    """
    Abstract base class for preprocessing defences implemented in PyTorch that support efficient preprocessor-chaining.
    """

    def __init__(self, device_type: str = "gpu", **kwargs):
        import torch

        super().__init__(**kwargs)

        # Set device
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:  # pragma: no cover
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device(f"cuda:{cuda_idx}")

    @abc.abstractmethod
    def forward(self, x: "torch.Tensor", y: Optional[Any] = None) -> Tuple["torch.Tensor", Optional[Any]]:
        """
        Perform data preprocessing in PyTorch and return preprocessed data as tuple.

        :param x: Dataset to be preprocessed.
        :param y: Labels to be preprocessed.
        :return: Preprocessed data.
        """
        raise NotImplementedError

    def estimate_forward(self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        """
        Provide a differentiable estimate of the forward function, so that autograd can calculate gradients
        of the defence for the backward pass. If the defence is differentiable, just call `self.forward()`.
        If the defence is not differentiable and a differentiable estimate is not available, replace with
        an identity function.

        :param x: Dataset to be preprocessed.
        :param y: Labels to be preprocessed.
        :return: Preprocessed data.
        """
        return self.forward(x, y=y)[0]

    @property
    def device(self):
        """
        Type of device on which the classifier is run, either `gpu` or `cpu`.
        """
        return self._device

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply preprocessing to input `x` and labels `y`.

        :param x: Sample to smooth with shape `(batch_size, width, height, depth)`.
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Smoothed sample.
        """
        import torch

        x_tensor = torch.tensor(x, device=self.device)
        if y is not None:
            y_tensor: Optional[torch.Tensor] = torch.tensor(y, device=self.device)
        else:
            y_tensor = None

        with torch.no_grad():
            x_tensor, y_tensor = self.forward(x_tensor, y_tensor)

        x_result = x_tensor.cpu().numpy()
        if y_tensor is not None:
            y_result: Optional[np.ndarray] = y_tensor.cpu().numpy()
        else:
            y_result = None
        return x_result, y_result

    # Backward compatibility.
    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        import torch

        def get_gradient(x, grad):
            x = torch.tensor(x, device=self.device, requires_grad=True)
            grad = torch.tensor(grad, device=self.device)

            x_prime = self.estimate_forward(x)
            x_prime.backward(grad)
            x_grad = x.grad.detach().cpu().numpy()

            if x_grad.shape != x.shape:
                raise ValueError(f"The input shape is {x.shape} while the gradient shape is {x_grad.shape}")

            return x_grad

        if x.dtype == object:
            x_grad_list = []
            for i, x_i in enumerate(x):
                x_grad_list.append(get_gradient(x=x_i, grad=grad[i]))
            x_grad = np.empty(x.shape[0], dtype=object)
            x_grad[:] = list(x_grad_list)
        elif x.shape == grad.shape:
            x_grad = get_gradient(x=x, grad=grad)
        else:
            # Special case for loss gradients
            x_grad = np.zeros_like(grad)
            for i in range(grad.shape[1]):
                x_grad[:, i, ...] = get_gradient(x=x, grad=grad[:, i, ...])

        return x_grad


class FeatureSqueezing(Preprocessor):
    """
    Reduces the sensibility of the features of a sample.

    | Paper link: https://arxiv.org/abs/1704.01155

    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general,
        see https://arxiv.org/abs/1902.06705
    """

    params = ["clip_values", "bit_depth"]

    def __init__(
        self,
        clip_values: CLIP_VALUES_TYPE,
        bit_depth: int = 8,
        apply_fit: bool = False,
        apply_predict: bool = True,
    ) -> None:
        """
        Create an instance of feature squeezing.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :param bit_depth: The number of bits per channel for encoding the data.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.clip_values = clip_values
        self.bit_depth = bit_depth
        self._check_params()

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply feature squeezing to sample `x`.

        :param x: Sample to squeeze. `x` values are expected to be in the data range provided by `clip_values`.
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Squeezed sample.
        """
        x_normalized = x - self.clip_values[0]
        x_normalized = x_normalized / (self.clip_values[1] - self.clip_values[0])

        max_value = np.rint(2 ** self.bit_depth - 1)
        res = np.rint(x_normalized * max_value) / max_value

        res = res * (self.clip_values[1] - self.clip_values[0])
        res = res + self.clip_values[0]

        return res, y

    def _check_params(self) -> None:
        if not isinstance(self.bit_depth, int) or self.bit_depth <= 0 or self.bit_depth > 64:
            raise ValueError("The bit depth must be between 1 and 64.")

        if len(self.clip_values) != 2:
            raise ValueError("`clip_values` should be a tuple of 2 floats containing the allowed data range.")

        if np.array(self.clip_values[0] >= self.clip_values[1]).any():
            raise ValueError("Invalid `clip_values`: min >= max.")