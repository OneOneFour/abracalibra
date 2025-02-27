from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch.distributions import Distribution
import torch.distributions.constraints
from .forward_model import ForwardModel


class Calibration(ABC):
    """
    Calibration class for defining the objectives of a calibration problem.
    Class can be inherited by various calibration objectives to actually undertake the calibration.

    Calibration determines the set of parameters that best fit a model to the data.
    Mathematically this finds the set of parameters, x, such that a forward model of the data:
    $$
    y = f(x) + \eta
    $$
    agrees with observations z = y + \epsilon within some tolerance.

    TODO:
    - Currently targets are fixed values, in general they may be functions of the model input
    - Currently we assume independence of forward model outputs, in general they may be correlated
    """

    def __init__(
        self,
        parameters: Union[int, Iterable],
        priors: Union[List[Distribution], Dict[Any, Distribution]],
        observables: Union[List[Distribution], Distribution, torch.Tensor],
        forward: Optional[ForwardModel] = None,
    ):
        """
        Initialize the calibration object.
        Parameters:
        -----------
        parameters : Union[int, np.ndarray, torch.Tensor]
            The parameters for the calibration. If an integer is provided, it will be converted to a range of parameters.
        priors : Union[Iterable, Dict]
            The priors for the parameters. Must be an iterable or a dictionary. If a dictionary is provided, its keys must match the parameters.
        summary:
            The summary function that takes the model output and returns a summary statistic.
            This evaluates the agreement between the model output and the observations.
        forward : Optional[Callable], optional
            The forward model function, by default None.
        Raises:
        -------
        ValueError
            If the number of priors does not match the number of parameters.
            If the keys of the priors dictionary do not match the parameters.
        """
        if isinstance(parameters, Iterable):
            self.parameter_names = parameters
        else:
            self.parameter_names = np.arange(parameters, dtype=int)

        if len(priors) != len(self.parameter_names):
            raise ValueError("The number of priors must match the number of parameters")
        if isinstance(priors, dict):
            if priors.keys() != set(self.parameter_names):
                raise ValueError(
                    "The keys of the priors dict must match the parameters"
                )
        self.priors = priors
        self.observables = observables
        self.forward = forward

        #TODO: probably change this in future to let Forward output a Distribution directly if it produces one.
        # can use that as a training point for the calibration 
        self.output_mc_batchshape = None # output provides multiple samples for the same input 

        if self.forward.out_shape:
            if self.number_of_observables != self.forward.out_shape[-1]:
                raise ValueError(
                    "The number of observables must match the output shape of the forward model"
                )
            self.output_mc_batchshape = self.forward.out_shape[:-1]
            

    def sample_prior(
        self, param_size: Union[torch.Size, int] = 1, labeled: bool = False
    ):
        if isinstance(param_size, int):
            param_size = torch.Size([param_size])
        samples = torch.stack([p.sample(param_size) for p in self.prior_list], dim=-1)
        if labeled:
            samples = dict(zip(self.parameter_names, samples.unbind(-1)))
        return samples

    def prior_support(self) -> List[torch.distributions.constraints.Constraint]:
        return [prior.support for prior in self.prior_list]

    def prior_log_pdf(self, theta: torch.Tensor):
        # We assume independence of the priors, i.e that each parameters prior is independent of others
        stack = []
        for prior, theta_p in zip(self.prior_list, theta.unbind(-1)):
            log_probs = torch.full(
                theta_p.shape, float("-inf"), dtype=theta_p.dtype, device=theta_p.device
            )
            valid_mask = prior.support.check(theta_p)
            log_probs[valid_mask] = prior.log_prob(theta_p[valid_mask])
            stack.append(log_probs)
        return torch.stack(stack, dim=-1).sum(-1)
        
        

    def kwargs_to_tensor(self, **kwargs):
        if set(kwargs.keys()) != set(self.parameter_names):
            intersection = set(kwargs.keys()).intersection(set(self.parameter_names))
            raise ValueError(
                "Keys of kwargs must match parameter names, missing keys: ",
                set(self.parameter_names) - intersection,
            )

        return torch.stack((kwargs[k] for k in self.parameter_names), dim=-1)

    def tensor_to_kwargs(self, tensor):
        if tensor.shape[-1] != len(self.parameter_names):
            raise ValueError(
                "Tensor must have the same number of parameters as the calibration object"
            )
        kwargs = {
            param_name: param
            for param_name, param in zip(self.parameter_names, tensor.unbind(dim=-1))
        }
        return kwargs

    @property
    def prior_list(self):
        if isinstance(self.priors, dict):
            return [self.priors[p] for p in self.parameter_names]
        else:
            return self.priors

    @property
    def prior_dict(self):
        if isinstance(self.priors, dict):
            return self.priors
        else:
            return dict(zip(self.parameter_names, self.priors))

    @property
    def number_of_parameters(self):
        return len(self.parameter_names)

    @property
    def number_of_observables(self):
        if isinstance(self.observables, Iterable):
            return len(self.observables)
        if isinstance(self.observables, Distribution):
            if self.observables.event_shape:
                return self.observables.event_shape[0]
            else:
                return 1
        raise ValueError("Observables must be a list of observables or a distribution")

    @abstractmethod
    def step(self, observations: torch.Tensor):
        """
        Perform a calibration step. Takes in new observations
        """
        raise NotImplementedError


class ApproxBayesianMethod(Calibration, ABC):
    def __init__(
        self,
        parameters: Union[int, np.ndarray, torch.Tensor],
        priors: Union[Iterable, Dict],
        observables: Union[List[Distribution], Distribution, torch.Tensor],
        summary: Callable[
            [Union[torch.Tensor, Distribution], Union[torch.Tensor, Distribution]],
            torch.Tensor,
        ],
        forward: Optional[Callable] = None,
    ):
        super(ApproxBayesianMethod, self).__init__(
            parameters, priors, observables, forward
        )
        self._summary = summary

    def summary(self, y: Union[torch.Tensor, Distribution]):
        return self._summary(y, self.observables)
