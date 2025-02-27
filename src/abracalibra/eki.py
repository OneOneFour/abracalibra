from .calibration import Calibration
from .lhs import LatinHypercubeSampler
from typing import Optional
from torch.distributions import (
    Distribution,
    MultivariateNormal,
    Normal,
    Uniform,
    TransformedDistribution,
)
from torch.distributions.transforms import SigmoidTransform, AffineTransform
from gpytorch.distributions import MultivariateNormal as GPyMultivariateNormal
from .misc import make_normal_transform
import torch


class EnsembleKalmanInversion(Calibration):
    """
    Ensemble Kalman Inversion, following process documented
    Inspired by Julia codebase
    """

    def __init__(
        self, parameters, priors, observables, ensemble_size, dt=1, forward=None
    ):
        super().__init__(parameters, priors, observables, forward)
        self.ensemble_size = ensemble_size
        self.iteration = 0

        self.results = torch.empty((0, self.ensemble_size, self.number_of_observables))
        self.__params = torch.empty((0, self.ensemble_size, self.number_of_parameters))
        self.times = torch.zeros((1,))
        self.dt = dt  # TODO: Learning rate schedule

        ## Convert all priors into normal (if not already)
        self.normal_priors = []  
        for p in self.prior_list:
            if isinstance(p, Normal):
                self.normal_priors.append(p)
            elif isinstance(p, TransformedDistribution):
                if isinstance(p.base_dist, Normal):
                    self.normal_priors.append(p.base_dist)
                else:
                    raise ValueError(
                        "If using transformed distribution priors, base must be a normal distribution."
                    )
            elif isinstance(p, Uniform):
                self.normal_priors.append(Normal(0, 1))
            else:
                raise ValueError(
                    "Only Normal and Uniform priors are supported currently. If you wish to use another distribution, try constructing it from a normal using a TransformedDistribution"
                )

        self.transforms = make_normal_transform(self.prior_list)

    @property
    def raw_parameters(self):
        return self.__params[: self.iteration+1]
    
    @property
    def parameters(self):
        return self.transform_params(self.__params[: self.iteration+1])


    def calibrate(self, n_steps):
        if self.forward is None:
            raise ValueError("Forward model must be provided")

        for params in self.eki_iter(n_steps):
            results = self.forward(params)
            next_params = self.step(results)

        return next_params.mean(dim=0)

    def transform_params(self, params: torch.Tensor):
        return torch.stack(
            [t(p) if t else p for t, p in zip(self.transforms, params.unbind(dim=-1))],
            dim=-1,
        )

    def inverse_transform_params(self, params: torch.Tensor):
        return torch.stack(
            [
                t.inv(p) if t else p
                for t, p in zip(self.transforms, params.unbind(dim=-1))
            ],
            dim=-1,
        )

    def generate_initial_ensemble(self, size: Optional[torch.Size] = None):
        if size is None:
            size = self.ensemble_size
        lhs = LatinHypercubeSampler(self.prior_list)
        ensemble = lhs.sample(size)
        return self.inverse_transform_params(ensemble)

    @property
    def observable_mean(self):
        if isinstance(self.observables, Distribution):
            return self.observables.mean
        if isinstance(self.observables, torch.Tensor):
            return self.observables
        means = []
        for obs in self.observables:
            if isinstance(obs, Distribution):
                means.append(obs.mean)
        return torch.stack(means, dim=0)

    @property
    def observable_covariance(self):
        if isinstance(self.observables, MultivariateNormal) or isinstance(
            self.observables, GPyMultivariateNormal
        ):
            return self.observables.covariance_matrix

        if isinstance(self.observables, Distribution):
            ## Assume independence.
            return torch.diag(self.observables.variance)
        if isinstance(self.observables, torch.Tensor):
            return torch.eye(self.observables.shape[-1]) * 1e-5
        variances = []
        ## Assuming independent observables if they are not provided as one
        for obs in self.observables:
            if isinstance(obs, Distribution):
                variances.append(obs.variance)
        return torch.diag(torch.stack(variances, dim=0))

    def step(self, results):
        if self.__params.shape[0] > self.iteration + 1:
            raise RuntimeError("Already invoked step for the current iteration")
        params = self.__params[self.iteration]  # n x d

        ens_residual = results - results.mean(dim=0)  # n x o 
        Cff = (ens_residual).T@(ens_residual)/self.ensemble_size
        residual =  self.observable_mean - results #  n x o
        # obs inv cov  is o x o
        inv_cov = torch.inverse(self.observable_covariance + Cff)
        Cparamsf = ( ens_residual.T @(params - params.mean(dim=0)))/(self.ensemble_size)

        params_update = residual @ inv_cov @ Cparamsf
        next_params = params + params_update
        self.__params = torch.cat((self.__params, next_params.unsqueeze(0)), dim=0)
        return self.transform_params(next_params)

    @property
    def current_time(self):
        return self.times[self.iteration]

    def eki_iter(self, n_steps):
        if self.iteration == 0:
            ensemble = self.generate_initial_ensemble()
            self.__params = ensemble.unsqueeze(0)
        for i in range(n_steps):
            self.iteration = i
            try:
                yield self.transform_params(self.__params[i])
            except KeyError as execep:
                raise RuntimeError(
                    "Attempted to get next iteration EKI params without first calling step().\n"
                    "Please call EnsembleKalmanInversion.step() with new params inside the loop"
                ) from execep
            self.times = torch.cat(
                [self.times, torch.tensor([self.times[-1] + self.dt])], dim=0
            )
            self.iteration = i + 1 
