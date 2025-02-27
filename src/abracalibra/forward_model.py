import torch
import inspect
from typing import Callable, Literal, Optional


class ForwardModel:
    def __init__(
        self,
        function: Callable,
        vectorized: bool = False,
        params_as: Literal["vector", "args", "kwargs"] = "kwargs",
        auto_broadcast_data: bool = True,
        out_shape: Optional[torch.Size] = None,
        **bind_data,
    ):
        self.__call_count = 0
        self.__function = function
        self.__vectorized = vectorized
        self.__params_as = params_as
        self.__bind_data = dict()
        self.__bind_data = self.__get_bind(**bind_data)
        self.__param_names = list(inspect.signature(function).parameters.keys())
        self.__auto_broadcast_data = auto_broadcast_data
        self.__out_shape = out_shape
        for k in bind_data.keys():
            try:
                self.__param_names.remove(k)
            except ValueError as exc:
                raise ValueError(
                    f"Bound data {k} not found in original function signature"
                ) from exc

    def __get_bind(self, **data):
        if len(data) == 0:  
            return dict()
        if self.__params_as != "kwargs":
            raise ValueError("Data can only be bound to kwargs models")
        new_data = {**self.__bind_data, **data}
        shapes = [
            v.shape if isinstance(v, torch.Tensor) else torch.tensor(v).shape
            for v in new_data.values()
        ]
        broadcast_shape = torch.broadcast_shapes(
            *shapes
        )  ## throws error if not compatible
        return {
            k: (
                v.expand(broadcast_shape)
                if isinstance(v,torch.Tensor)
                else torch.tensor(v).expand(broadcast_shape)
            )
            for k, v in new_data.items()
        }

    def bind(self, **data):
        return ForwardModel(
            self.__function,
            self.__vectorized,
            params_as="kwargs",
            auto_broadcast_data=self.__auto_broadcast_data,
            **self.__get_bind(**data),
        )

    @property
    def param_names(self):
        return frozenset(self.__param_names)

    @property
    def function(self):
        return self.__function

    def bound_data(self, batch_shape=None):
        if self.__auto_broadcast_data:
            return {
                k: v.expand(*batch_shape, *v.shape) for k, v in self.__bind_data.items()
            }
        else:
            return {k: v for k, v in self.__bind_data.items()}

    @property
    def data_shape(self):
        shape = list(self.__bind_data.values())[0].shape
        return shape

    @property
    def bound_data_nonvec_generator(self):
        return (
            {k: v for k, v in zip(self.__bind_data.keys(), item)}
            for item in zip(*[v.ravel() for v in self.__bind_data.values()])
        )

    @property
    def out_shape(self):
        if self.__out_shape:
            return self.__out_shape
        return torch.Size([-1])

    @property
    def call_count(self):
        return self.__call_count

    def __call__(self, x) -> torch.Tensor:
        """
        Evaluates the function with the given parameters in tensor `x`.

        Parameters:
        x (torch.Tensor): A tensor containing the parameters for the function. The shape and interpretation of `x` depend on the value of `self.__params_as`.

        Returns:
        torch.Tensor: The result of applying the function to the parameters in `x`.

        The behavior of this method depends on the value of `self.__params_as`:
        - "vector": `x` is treated as a vector of parameters. If `self.__vectorized` is False, the function is applied to each parameter vector individually and the results are stacked. If `self.__vectorized` is True, the function is applied to `x` directly.
        - "args": `x` is treated as a set of positional arguments. If `self.__vectorized` is False, the function is applied to each set of arguments individually and the results are stacked. If `self.__vectorized` is True, the function is applied to `x` directly using unpacking.
        - "kwargs": `x` is treated as a set of keyword arguments. If `self.__vectorized` is False, the function is applied to each set of keyword arguments individually and the results are stacked. If `self.__vectorized` is True, the function is applied to `x` directly using unpacking.

        Note:
        - `self.__function` is the function to be evaluated.
        - `self.__param_names` is a list of parameter names used when `self.__params_as` is "kwargs".
        - `self.__vectorized` indicates whether the function can be applied to the entire tensor at once.
        """
        self.__call_count += 1
        if self.__params_as == "vector":
            if not self.__vectorized:
                params = x.reshape(-1, x.shape[-1])
                return (
                    torch.stack([self.__function(p) for p in params])
                    .view(*x.shape[:-1], -1)
                    .squeeze(-1)
                )

            return self.__function(x, **self.bound_data(x.shape[:-1]))
        if self.__params_as == "args":
            if not self.__vectorized:
                params = x.reshape(-1, x.shape[-1])
                return (
                    torch.stack([self.__function(*p) for p in params])
                    .view(*x.shape[:-1], *self.out_shape)
                    .squeeze(-1)
                )

            return self.__function(*x.unbind(dim=-1), **self.bound_data(x.shape[:-1]))
        if self.__params_as == "kwargs":
            if not self.__vectorized:
                param_dicts = (
                    {k: v for k, v in zip(self.__param_names, item)}
                    for item in x.view(-1, len(self.__param_names))
                )
                if self.__bind_data:
                    total_data = []
                    for p in param_dicts:
                        total_data.append(
                            torch.stack(
                                [
                                    self.__function(**p, **data)
                                    for data in self.bound_data_nonvec_generator
                                ]
                            ).view(*self.data_shape, *self.out_shape)
                        )
                    return (
                        torch.stack(total_data)
                        .view(*x.shape[:-1], *self.data_shape, *self.out_shape)
                        .squeeze(-1)
                    )
                else:
                    return (
                        torch.stack([self.__function(**p) for p in param_dicts])
                        .view(*x.shape[:-1], *self.out_shape)
                        .squeeze(-1)
                    )

            return self.__function(
                **{
                    k: (
                        v.view(*x.shape[:-1], *([1] * len(self.data_shape)))
                        if self.__bind_data
                        else v
                    )
                    for k, v in zip(self.__param_names, x.unbind(dim=-1))
                },
                **self.bound_data(x.shape[:-1]),
            )

    def __repr__(self):
        return f"ForwardModel({self.__function.__name__}, vectorized={self.__vectorized}, params_as={self.__params_as}, bind_data={self.__bind_data})"
