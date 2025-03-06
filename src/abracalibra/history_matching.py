from collections.abc import Iterable
from typing import Callable, Dict, List, Optional, Union
import warnings
import numpy as np
import torch
import torch.distributions.constraints
import torch.nn as nn
import gpytorch
from torch.distributions import Distribution, Uniform
from torch.distributions.constraints import _Interval as Interval

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import MultitaskGaussianLikelihood

from .calibration import ApproxBayesianMethod
from .lhs import LatinHypercubeSampler
from .emulators import MultiTaskGP
from .forward_model import ForwardModel
from abc import ABC, abstractmethod


class HistoryMatchingBase(ApproxBayesianMethod, ABC):
    def __init__(
        self,
        parameters: Union[int, np.ndarray, torch.Tensor],
        priors: Union[Iterable, Dict],
        observables: Union[List[Distribution], Distribution, torch.Tensor],
        summary: Callable,
        ensemble_size: int,
        implausibility_cutoff: float = 3.0,
        forward: Optional[ForwardModel] = None,
    ):
        super().__init__(parameters, priors, observables, summary, forward)
        self.implausibility_cutoff = implausibility_cutoff
        self.ensemble_size = ensemble_size
        self.__emulators = nn.ModuleList([])
        self.__params = None
        self.__results = None
        self.__current_wave = 0

    @property
    def emulators(self):
        # Take care to note mutate the emulators or else some undefined behaviour might happen.
        return nn.ModuleList([e for e in self.__emulators if e is not None])

    @property
    def params(self):
        if self.__params is None:
            return torch.zeros((0, self.ensemble_size, self.number_of_parameters))
        return self.__params[: self.current_wave]

    @property
    def results(self):
        if self.__params is None:
            if self.output_mc_batchshape:
                return torch.zeros(
                    (
                        0,
                        self.ensemble_size,
                        *self.output_mc_batchshape,
                        self.number_of_observables,
                    )
                )
            return torch.zeros((0, self.ensemble_size, self.number_of_observables))
        return self.__results[: self.current_wave]

    def calibrate(
        self, nwaves: int, early_stop_threshold: float = -1, verbose: bool = False
    ):
        if self.forward is None:
            raise RuntimeError("Forward model must be provided to calibrate directly")

        for wave, params in self.hm_iter(
            nwaves, early_stop_threshold=early_stop_threshold, verbose=verbose
        ):
            if verbose:
                print(f"Working on wave {wave}")
            ## Update parameter array

            results = self.forward(params)
            ## Build parameters
            self.step(results, wave)

    def implausibility(
        self, theta: torch.Tensor, wave: Optional[int] = None
    ) -> torch.Tensor:
        if wave is None:
            wave = self.current_wave
        theta_org_shape = theta.shape[:-1]
        flt_theta = theta.view(-1, self.number_of_parameters)
        batches = []
        print(f"torch.no_grad() active: {not torch.is_grad_enabled()}")
        print(
            f"gpytorch.settings.fast_pred_var active: {gpytorch.settings.fast_pred_var.on()}"
        )
        for i, batch in enumerate(torch.split(flt_theta, 1000)):
            print(f"Working on batch {i}", end="\r", flush=True)
            emulator = self.emulators[wave]
            # likelihood = self.emulators[wave].likelihood 
            pred_y = emulator(batch)
            imp = self.summary(pred_y)
            batches.append(imp)
        
        imp = torch.cat(batches, dim=0)
        if theta_org_shape:
            return imp.view(*theta_org_shape)
        else:
            return imp

    def hm_iter(
        self, nwaves: int, early_stop_threshold: float = -1, verbose: bool = False
    ):
        """
        Perform history matching iterations.

        This generator function performs history matching over a specified number of waves.
        It initializes the emulators, parameters, and results if it's the first wave, and
        iterates through each wave, checking for convergence and yielding the parameters
        for each wave.

        Args:
            nwaves (int): The number of waves to perform history matching.

        Yields:
            tuple: A tuple containing the current wave index and the parameters for that wave.

        Notes:
            - If the current wave is 0, the function initializes the emulators, parameters,
              and results.
            - The function checks for convergence using the `check_if_converged` method and
              breaks the loop if convergence is achieved.
            - The `get_params` method is used to retrieve the parameters for each wave.
        """
        if self.__current_wave == 0:
            self.__emulators = nn.ModuleList([])
            self.__params = torch.empty(
                (nwaves, self.ensemble_size, self.number_of_parameters)
            )
            if self.output_mc_batchshape:
                self.__results = torch.empty(
                    (nwaves, self.ensemble_size, *self.output_mc_batchshape, self.number_of_observables)
                )
            else:
                self.__results = torch.empty(
                    (nwaves, self.ensemble_size, self.number_of_observables)
                )

        for wave in range(self.current_wave, nwaves):
            # Reset this __just in case__ it was changed by mistake.
            self.__current_wave = wave
            # if self.check_if_converged(early_stop_threshold,verbose=verbose):
            #     break
            wave_params = self.sample(wave, verbose=verbose)
            self.__params[wave] = wave_params
            yield wave, wave_params
            # This is needed to have the correct __current_wave value on the final iteration.
            self.__current_wave = wave + 1

    def generate_initial_ensemble(self, size: Optional[int] = None) -> torch.Tensor:
        """
        Generates the initial ensemble using Latin Hypercube Sampling (LHS).

        Parameters:
        size (Optional[int]): The size of the ensemble to generate. If not provided,
                              the default ensemble size will be used.

        Returns:
        torch.Tensor: A tensor containing the generated ensemble.
        """
        if size is None:
            size = self.ensemble_size

        lhs = LatinHypercubeSampler(self.prior_list)
        ensemble = lhs.sample(size)
        return ensemble

    @property
    def current_wave(self):
        return self.__current_wave

    def sample(
        self,
        wave: Optional[int] = None,
        size: Optional[int] = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Sample potential parameters from current NROY space for wave.
        By default sample current wave and for the defined ensemble size.
        """

        if size is None:
            size = self.ensemble_size
        if wave is None:
            wave = self.current_wave
        if wave == 0:
            return self.generate_initial_ensemble(size)
        ## Create samples
        samples = torch.empty((size, self.number_of_parameters))
        if verbose:
            j = 0
            best_imp = torch.inf
        for i in range(size):

            while True:

                samples[i] = self.sample_prior()
                if verbose:
                    imp = self.implausibility(samples[i], wave - 1)
                    if imp < best_imp:
                        best_imp = imp
                    j += 1
                    print(
                        f"Sampled Prior {j} times, obtained {i} samples, best implausibility {best_imp.item()}",
                        end="\r",
                        flush=True,
                    )

                if self.nroy(samples[i], wave):
                    break
        if verbose:
            print("\n")
        return samples

    @abstractmethod
    def make_emulator(self, parameters, observations) -> nn.Module:
        raise NotImplementedError

    def step(self, observations, wave: Optional[int] = None):
        if wave is None:
            wave = self.current_wave
        # ingest wave
        params = self.__params[: wave + 1]
        self.__results[wave] = (
            observations.unsqueeze(1) if observations.dim() < 2 else observations
        )
        results = self.__results[: wave + 1]

        emulator = self.make_emulator(params, results)
        emulator.eval()
        self.__emulators.append(emulator)
        return emulator

    def nroy(self, theta, wave: Optional[int] = None) -> torch.Tensor:
        # Check if in prior space
        if wave is None:
            wave = self.current_wave
        base_nroy = ~torch.isclose(self.prior_log_pdf(theta).exp(), torch.tensor(0.0))
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(wave):
                imp = self.implausibility(theta, wave=i)
                base_nroy = base_nroy & (imp <= self.implausibility_cutoff)
        return base_nroy

    def get_support_axis(self, wave: Optional[int] = None, num_points: int = 100):
        num_points = int(num_points)
        if wave is None:
            wave = self.current_wave
        supports = self.prior_support()
        coord_axis = torch.empty((self.number_of_parameters, num_points))
        for i, support in enumerate(supports):
            if isinstance(support, Interval):
                coord_axis[i] = torch.linspace(
                    support.lower_bound, support.upper_bound, num_points
                )
            else:
                # if no support uses support for points that we have
                std = torch.std(self.params[..., i])
                coord_axis[i] = torch.linspace(
                    torch.min(self.params) - std,
                    torch.max(self.params) + std,
                    num_points,
                )
        return coord_axis

    def nroy_over_support(
        self, wave: Optional[int] = None, num_points: int = 100, return_coords=False
    ) -> torch.Tensor:
        """
        Returns the NROY Space over the support of the prior distribution.

        Returns nroy space, plus coordinate axis for the domain of the prior distribution.
        """
        coord_axis = self.get_support_axis(wave, num_points)
        grids = torch.meshgrid(*coord_axis, indexing="ij")

        # Flatten the grids
        flattened_grids = torch.stack([grid for grid in grids], dim=-1).view(
            -1, self.number_of_parameters
        )
        nroy = self.nroy(flattened_grids, wave).reshape(grids[0].shape)
        if return_coords:
            return nroy, coord_axis
        else:
            return nroy

    def estimate_nroy_volume(
        self, wave: Optional[int] = None, num_points: int = 100
    ) -> torch.Tensor:
        """
        Estimate the volume of the NROY space.
        """
        num_points = int(num_points)
        if wave is None:
            wave = self.current_wave
        nroy_space, coords = self.nroy_over_support(wave, return_coords=True)
        volume_element = torch.prod((coords[:, -1] - coords[:, 0]) / num_points)
        return torch.sum(nroy_space) * volume_element

    def check_if_converged(
        self, threshold: float, num_points: int = 100, verbose: bool = False
    ) -> bool:
        """
        Converged if NROY relative area change is below than threshold fraction.
        Involves estimating the NROY space
        """
        # This is invoked at the start of each wave
        # Before Wave 0 we have no information at all
        # Before Wave 1 we have no information on the relative area change
        if self.current_wave == 0:
            return False

        current_nroy, coords = self.nroy_over_support(
            num_points=num_points, return_coords=True
        )
        volume_element = torch.prod((coords[:, -1] - coords[:, 0]) / num_points)
        current_volume = torch.sum(current_nroy) * volume_element
        if verbose:
            print("Current Volume: ", current_volume)
        if torch.isclose(current_volume, torch.tensor(0.0)):
            warnings.warn(
                "Current NROY volume is very close to zero, finishing calibration"
            )
            return True
        if self.current_wave < 2:
            return False

        # Use supports to calculate bounding volume of prior space

        past_nroy = self.nroy_over_support(self.current_wave - 1, num_points=num_points)
        past_volume = torch.sum(past_nroy) * volume_element
        if verbose:

            print("Past Volume: ", past_volume)

        return (
            torch.abs(current_volume - past_volume) / past_volume < threshold
        ).item()


class HistoryMatchingOCE(HistoryMatchingBase, ABC):
    @property
    def current_emulator(self):
        return self.emulators[self.current_wave]

    def nroy(self, theta, wave: Optional[int] = None) -> torch.Tensor:
        # Check if in prior space
        if wave is None:
            wave = self.current_wave
        base_nroy = torch.isclose(self.prior_log_pdf(theta), torch.tensor(0.0))
        pred_y = self.current_emulator(theta.view(-1, self.number_of_parameters))
        imp = self.summary(pred_y)
        base_nroy = base_nroy & (imp <= self.implausibility_cutoff)
        return base_nroy


class HistoryMatchingSingleTaskGPLastWaveOnly(HistoryMatchingBase):
    def make_emulator(self, parameters, observations):
        # Only use current parameters + observations
        if self.current_wave > 0:
            valid_params = self.nroy(parameters[:-1], wave=self.current_wave)
            valid_old_params = parameters[:-1][valid_params, :]
            valid_old_obs = observations[:-1][valid_params, :]
            parameters = torch.cat([valid_old_params, parameters[-1]], dim=0)
            observations = torch.cat([valid_old_obs, observations[-1]], dim=0)
        parameters = parameters.view(-1, self.number_of_parameters)
        observations = observations.view(-1, self.number_of_observables)
        model = SingleTaskGP(
            parameters.double(), observations.double(), outcome_transform=None
        )
        model.train()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()
        return model


class HistoryMatchingSingleTaskGP(HistoryMatchingBase):
    """
    Simple wrapper for HistoryMatchingBase to use SingleTaskGP from BoTorch as the emulator.
    Other emulators can be defined in a similar manner.

    TODO: allow saving of models.

    """

    def make_emulator(self, parameters, observations) -> nn.Module:
        parameters = parameters.view(-1, self.number_of_parameters)

        observations = observations.view(-1, self.number_of_observables)
        model = SingleTaskGP(
            parameters.double(), observations.double(), outcome_transform=None
        )
        model.train()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()
        return model


class HistoryMatchingMultiTask(HistoryMatchingBase):
    def make_emulator(self, parameters, observations):
        if self.output_mc_batchshape:
            target_shape = torch.Size((*parameters.shape[:-1], *self.output_mc_batchshape, self.number_of_parameters))
            parameters = parameters.unsqueeze(-2).expand(*target_shape)
            parameters = parameters.reshape(-1, self.number_of_parameters)
        else:
            parameters = parameters.view(-1, self.number_of_parameters)
        observations = observations.view(-1, self.number_of_observables)
        likelihood = MultitaskGaussianLikelihood(
            num_tasks=self.number_of_observables, has_task_noise=False
        )
        # likelihood.noise = 1e-2
        model = MultiTaskGP(
            parameters,
            observations,
            likelihood=likelihood,
            num_tasks=self.number_of_observables,
        ).float()
        optim = torch.optim.Adam(model.parameters(), lr=0.05)
        mll = ExactMarginalLogLikelihood(likelihood, model)
        for i in range(250):
            optim.zero_grad()
            output = model(parameters)
            loss = -mll(output, observations)
            loss.backward()
            optim.step()
            print(f"Step {i} Loss: {loss.item()}")
        model.eval()
        return model
