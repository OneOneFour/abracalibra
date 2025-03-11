from abc import ABC, abstractmethod
from itertools import combinations_with_replacement
from typing import Callable

import torch
import torch.nn as nn


def get_polynomial_features(x: torch.Tensor, degree=2):
    n = x.shape[-1]
    terms = []
    for deg in range(1, degree + 1):
        terms.extend(combinations_with_replacement(range(n), deg))
    poly_features = []
    for t in terms:
        poly_features.append(torch.prod(x[..., t], dim=-1, keepdim=True))
    return torch.cat(poly_features, dim=-1)



def rk4_step(
    f: Callable[[torch.Tensor], torch.Tensor],
    t: float,
    y: torch.Tensor,
    dt: float,
    *args,
):
    k1 = f(t, y, *args)
    k2 = f(t + dt / 2, y + dt / 2 * k1, *args)
    k3 = f(t + dt / 2, y + dt / 2 * k2, *args)
    k4 = f(t + dt, y + dt * k3, *args)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class ODESystem(nn.Module, ABC):
    @abstractmethod
    def ode(self, t, r):
        
        pass

    def forward(self, T, dt,x0):
        """
        Integrate the ODE system from x0 to T with time step dt
        Returns a tensor of shape (len(t),3) where t is the time points
        """
        t = torch.arange(0, T, dt,dtype=x0.dtype)
        batch_shape = x0.shape[:-1]
        r = torch.cat(
            [
                x0.unsqueeze(-2),
                torch.zeros((*batch_shape, len(t) - 1, x0.shape[-1]),dtype=x0.dtype),
            ],
            dim=-2)
        for i, t_i in enumerate(t[:-1]):
            r[..., i + 1, :] = rk4_step(self.ode, t_i, r[..., i, :], dt)
        return r


class Lorenz63(ODESystem):
    def __init__(self, sigma, beta, rho, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = nn.Parameter(data=torch.as_tensor(sigma).double())
        self.beta = nn.Parameter(data=torch.as_tensor(beta).double())
        self.rho = nn.Parameter(data=torch.as_tensor(rho).double())

    def ode(self, t, r):
        x, y, z = r[..., 0], r[..., 1], r[..., 2]
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return torch.stack([dxdt, dydt, dzdt], dim=-1)


class Lorenz96OneLevel(ODESystem):
    def __init__(self, F, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.F = nn.Parameter(torch.as_tensor(F).double())

    def ode(self, t, r):
        return (
            (torch.roll(r, -1, dims=-1) - torch.roll(r, 2, dims=-1))
            * torch.roll(r, 1, dims=-1)
            - r
            + self.F
        )

class Lorenz96OneLevelScaled(ODESystem):
    def __init__(self,F,diag_linear_terms, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.F = F
        self.diag_linear_terms = diag_linear_terms



    def linear_terms(self,r):
        base = torch.diagflat(torch.ones(r.shape[-1])*-1)


    def ode(self,t,r):
        return (
            self.adv*(torch.roll(r, -1, dims=-1) - torch.roll(r, 2, dims=-1))
            * torch.roll(r, 1, dims=-1)
            - r@self.linear_terms(r)
            + self.F
        )


class Lorenz96TwoLevel(ODESystem):
    def __init__(self, F, b, h, c, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.F = nn.Parameter(torch.as_tensor(F).double())
        self.b = nn.Parameter(torch.as_tensor(b).double())
        self.h = nn.Parameter(torch.as_tensor(h).double())
        self.c = nn.Parameter(torch.as_tensor(c).double())

    def ode(self, t, xy, N, J):
        x = xy[..., :N]
        y = xy[..., N:]

        xdot = (
            (torch.roll(x, -1, dims=-1) - torch.roll(x, 2, dims=-1))
            * torch.roll(x, 1, dims=-1)
            - x
            + self.F
            - (self.h * self.c / self.b)
            * torch.sum(y.view(*y.shape[:-1], N, J), dim=-1)
        )
        ydot = (
            self.c
            * self.b
            * (torch.roll(y, 1, dims=-1) - torch.roll(y, -2, dims=-1))
            * torch.roll(y, -1, dims=-1)
            - self.c * y
            + (self.h * self.c / self.b) * x.repeat_interleave(J,dim=-1)
        )

        return torch.cat([xdot, ydot], dim=-1)

    def forward(self, T, dt,x0,y0):
        """
        Inputs:
        x0: Initial condition for the x variables, shape (batch_shape, N)
        y0: Initial condition for the y variables, shape (batch_shape, N*J)
        T: Final time
        dt: Time step
        """
        time = torch.arange(0, T, dt,dtype=x0.dtype)
        batch_shape = x0.shape[:-1]
        if y0.shape[:-2] != batch_shape:
            raise ValueError("The batch shapes of x0 and y0 must match")
        N = x0.shape[-1]
        if y0.shape[-2] != N:
            raise ValueError(
                "The number of y variables must be a multiple of the number of x variables"
            )
        J = int(y0.shape[-1])
        xy0 = torch.cat([x0, y0.view(-1,N*J)], dim=-1)  # xy0 has dimension (batch_shape, N+J*N)
        xy = torch.cat(
            [
                xy0.unsqueeze(-2),
                torch.zeros((*batch_shape, len(time) - 1, xy0.shape[-1])).double(),
            ],
            dim=-2,
        )

        for i, t in enumerate(time[:-1]):
            xy[..., i + 1, :] = rk4_step(self.ode, t, xy[..., i, :], dt, N, J)
        x = xy[..., :N]
        y = xy[..., N:].view(*batch_shape,len(time),N,J)
        return x,y

class TimeAveragedFeatures(nn.Module):
    def __init__(self, base:ODESystem, spinup:int,degree:int = 2,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base = base
        self.spinup = spinup
        self.degree = degree 

    def forward(self,T,dt,*args):
        x = self.base(T,dt,*args)
        if isinstance(x,tuple):
            return tuple(map(lambda v: torch.mean(get_polynomial_features(v[...,int(self.spinup/dt):,:],degree=self.degree),dim=-2),x))
        else:
            return torch.mean(get_polynomial_features(x[...,int(self.spinup/dt):,:],degree=self.degree),dim=-2)