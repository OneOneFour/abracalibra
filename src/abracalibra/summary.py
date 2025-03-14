from functools import wraps
from typing import Union

import torch
from gpytorch.distributions import MultivariateNormal as GPyMultivariateNormal
from gpytorch.distributions import MultitaskMultivariateNormal
from torch.distributions import Normal, MultivariateNormal

from .misc import get_task_covariance


def l2_norm(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum((y - z) ** 2, dim=-1))


def l1_norm(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return torch.abs(torch.sum(y - z, dim=-1))


def linf_norm(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.abs(y - z), dim=-1).values


def ln_norm(y: torch.Tensor, z: torch.Tensor, n: int) -> torch.Tensor:
    return torch.sum(torch.abs(y - z) ** n, dim=-1) ** (1 / n)


def implausibility_univariate_distr(
    y: Union[Normal, MultivariateNormal, GPyMultivariateNormal], z: Normal
):
    return torch.abs(y.mean - z.mean) / torch.sqrt(y.variance+ z.variance.squeeze())


def batched_implausibility_univariate_distr(
    y: Union[Normal, MultivariateNormal, GPyMultivariateNormal], z: Normal
):
    """
    If y, z are both batched normal distributions, that is for example in the GP
    case where outputs are assumed independent then this method works.

    Output will have shape n x d  where n is the number of samples and d is the number of outputs
    """
    y_mean_T = torch.permute(y.mean,(*range(1,y.mean.dim()),0))
    z_mean_T = z.mean.view(*[1 for _ in range(y.mean.dim()-1)],-1)
    y_var_T = torch.permute(y.variance,(*range(1,y.mean.dim()),0))
    z_var_T = z.variance.view( *[1 for _ in range(y.variance.dim()-1)],-1)
    return torch.abs(y_mean_T - z_mean_T) / torch.sqrt(
        y_var_T + z_var_T
    )


def implausibility_multivariate_distr(
    y: MultitaskMultivariateNormal, z: Union[MultivariateNormal, GPyMultivariateNormal]
):
    npoints, ntasks = y.event_shape
    if ntasks != z.event_shape[0]:
        raise ValueError(
            "Number of tasks must match number of observations!\n"
            f"Got {ntasks} with a z event shape of {z.event_shape}"
        )
    covar = get_task_covariance(y) + z.covariance_matrix.unsqueeze(0)
    offset = y.mean - z.mean
    prod = torch.bmm(
        offset.unsqueeze(-2), covar.inverse()
    )  # TODO: See if torch.linalg.solve is available consistently on all devices!
    quad = torch.bmm(prod, offset.unsqueeze(-1)).squeeze(dim=[-2, -1])
    return quad


def set_truth(fn, true_obs):
    @wraps(fn)
    def wrapper(y):
        return fn(y, z=true_obs)

    return wrapper
