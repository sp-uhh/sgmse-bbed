"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

Taken and adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py
"""
import abc
import warnings
import math
import scipy.special as sc
import numpy as np
from sgmse.util.tensors import batch_broadcast
import torch

from sgmse.util.registry import Registry


SDERegistry = Registry("SDE")


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
            N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t, *args):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t, *args):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x|args)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape, *args):
        """Generate one sample from the prior distribution, $p_T(x|args)$ with shape `shape`."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
            z: latent code
        Returns:
            log probability density
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def add_argparse_args(parent_parser):
        """
        Add the necessary arguments for instantiation of this SDE class to an argparse ArgumentParser.
        """
        pass

    def discretize(self, x, t, y, stepsize):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
            f, G
        """
        dt = stepsize
        #dt = 1 /self.N
        drift, diffusion = self.sde(x, t, y)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(oself, score_model, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
            score_model: A function that takes x, t and y and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = oself.N
        T = oself.T
        sde_fn = oself.sde
        discretize_fn = oself.discretize

        # Build the class for reverse-time SDE.
        class RSDE(oself.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t, *args):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                rsde_parts = self.rsde_parts(x, t, *args)
                total_drift, diffusion = rsde_parts["total_drift"], rsde_parts["diffusion"]
                return total_drift, diffusion

            def rsde_parts(self, x, t, *args):
                sde_drift, sde_diffusion = sde_fn(x, t, *args)
                score = score_model(x, t, *args)
                score_drift = -sde_diffusion[:, None, None, None]**2 * score * (0.5 if self.probability_flow else 1.)
                diffusion = torch.zeros_like(sde_diffusion) if self.probability_flow else sde_diffusion
                total_drift = sde_drift + score_drift
                return {
                    'total_drift': total_drift, 'diffusion': diffusion, 'sde_drift': sde_drift,
                    'sde_diffusion': sde_diffusion, 'score_drift': score_drift, 'score': score,
                }

            def discretize(self, x, t, y, stepsize):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t, y, stepsize)
                if torch.is_complex(G):
                    G = G.imag

                rev_f = f - G[:, None, None, None] ** 2 * score_model(x, t, y) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()

    @abc.abstractmethod
    def copy(self):
        pass


@SDERegistry.register("ouve")
class OUVESDE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--sde-n", type=int, default=1000, help="The number of timesteps in the SDE discretization. 30 by default")
        parser.add_argument("--theta", type=float, default=1.5, help="The constant stiffness of the Ornstein-Uhlenbeck process. 1.5 by default.")
        parser.add_argument("--sigma-min", type=float, default=0.05, help="The minimum sigma to use. 0.05 by default.")
        parser.add_argument("--sigma-max", type=float, default=0.5, help="The maximum sigma to use. 0.5 by default.")
        return parser

    def __init__(self, theta, sigma_min, sigma_max, N=1000, **ignored_kwargs):
        """Construct an Ornstein-Uhlenbeck Variance Exploding SDE.

        Note that the "steady-state mean" `y` is not provided at construction, but must rather be given as an argument
        to the methods which require it (e.g., `sde` or `marginal_prob`).

        dx = -theta (y-x) dt + sigma(t) dw

        with

        sigma(t) = sigma_min (sigma_max/sigma_min)^t * sqrt(2 log(sigma_max/sigma_min))

        Args:
            theta: stiffness parameter.
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N)
        self.theta = theta
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.logsig = np.log(self.sigma_max / self.sigma_min)
        self.N = N
        self._T = 1

    def copy(self):
        return OUVESDE(self.theta, self.sigma_min, self.sigma_max, N=self.N)

    @property
    def T(self):
        return self._T

    def sde(self, x, t, y):
        drift = self.theta * (y - x)
        # the sqrt(2*logsig) factor is required here so that logsig does not in the end affect the perturbation kernel
        # standard deviation. this can be understood from solving the integral of [exp(2s) * g(s)^2] from s=0 to t
        # with g(t) = sigma(t) as defined here, and seeing that `logsig` remains in the integral solution
        # unless this sqrt(2*logsig) factor is included.
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * np.sqrt(2 * self.logsig)
        return drift, diffusion

    def _mean(self, x0, t, y):
        theta = self.theta
        exp_interp = torch.exp(-theta * t)[:, None, None, None]
        return exp_interp * x0 + (1 - exp_interp) * y

    def _std(self, t):
        # This is a full solution to the ODE for P(t) in our derivations, after choosing g(s) as in self.sde()
        sigma_min, theta, logsig = self.sigma_min, self.theta, self.logsig
        # could maybe replace the two torch.exp(... * t) terms here by cached values **t
        return torch.sqrt(
            (
                sigma_min**2
                * torch.exp(-2 * theta * t)
                * (torch.exp(2 * (theta + logsig) * t) - 1)
                * logsig
            )
            /
            (theta + logsig)
        )

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(torch.ones((y.shape[0],), device=y.device))
        z = torch.randn_like(y)
        #z = torch.zeros_like(y)
        x_T = y + z * std[:, None, None, None]
        return x_T, z

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for OU SDE not yet implemented!")



#This is the SDE for the Brownian Bridge with Exploding Diffusion Coefficient. It uses the same parameterization as in the paper.
@SDERegistry.register("bbed")
class BBED(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--sde-n", type=int, default=30, help="The number of timesteps in the SDE discretization. 30 by default")
        parser.add_argument("--T_sampling", type=float, default=0.999, help="The T so that t < T during sampling in the train step.")
        parser.add_argument("--k", type=float, default = 2.6, help="base factor for diffusion term") 
        parser.add_argument("--theta", type=float, default = 0.52, help="root scale factor for diffusion term.")
        return parser

    def __init__(self, T_sampling, k, theta, N=1000, **kwargs):
        """Construct an Brownian Bridge with Exploding Diffusion Coefficient SDE with parameterization as in the paper.
        dx = (y-x)/(Tc-t) dt + sqrt(theta)*k^t dw
        """
        super().__init__(N)
        self.k = k
        self.logk = np.log(self.k)
        self.theta = theta
        self.N = N
        self.Eilog = sc.expi(-2*self.logk)
        self.T = T_sampling #for sampling in train step and inference
        self.Tc = 1 #for constructing the SDE, dont change this


    def copy(self):
        return BBED(self.T, self.k, self.theta, N=self.N)


    def T(self):
        return self.T
    
    def Tc(self):
        return self.Tc


    def sde(self, x, t, y):
        drift = (y - x)/(self.Tc - t)
        sigma = (self.k) ** t
        diffusion = sigma * np.sqrt(self.theta)
        return drift, diffusion


    def _mean(self, x0, t, y):
        time = (t/self.Tc)[:, None, None, None]
        mean = x0*(1-time) + y*time
        return mean

    def _std(self, t):
        t_np = t.cpu().detach().numpy()
        Eis = sc.expi(2*(t_np-1)*self.logk) - self.Eilog
        h = 2*self.k**2*self.logk
        var = (self.k**(2*t_np)-1+t_np) + h*(1-t_np)*Eis
        var = torch.tensor(var).to(device=t.device)*(1-t)*self.theta
        return torch.sqrt(var)

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(self.T*torch.ones((y.shape[0],), device=y.device))
        z = torch.randn_like(y)
        x_T = y + z * std[:, None, None, None]
        return x_T, z

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for BBED not yet implemented!")
    



#This is the old class. It uses an equivalent representation as in the paper.
# Instead of k, the class below uses sigma_min and sigma_max, so that k = sigma_max/sigma_min.
""" 
@SDERegistry.register("bbve")
class BBVE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--sde-n", type=int, default=30, help="The number of timesteps in the SDE discretization. 30 by default")
        parser.add_argument("--T_sampling", type=float, default=0.999, help="The T so that t < T during sampling in the train step.")
        parser.add_argument("--sigma-min", type=float, default=1, help="The minimum sigma to use. Set it to 1 and dont change it.")
        parser.add_argument("--sigma-max", type=float, default=0.5, help="This is k, the base of diffusion term, when sigma min is 1.")
        parser.add_argument("--theta", type=float, default=0.53, help="This rescales the diffusion term")
        return parser

    def __init__(self, T_sampling, sigma_min, sigma_max, theta, N=1000, **kwargs):
        #Construct an Brownian Bridge with Exploding Diffusion Coefficient SDE version 2.
        #This is the a equivalent version of the BBED SDE with a different parameterization.
        #To obtain same parameterization as above one must set sigma_min = 1 and sigma_max = k.
        #dx = (y-x)/(Tc-t) dt + sqrt(theta)*(sigmamax/sigmamin)^t dw
        
        super().__init__(N)
        self.sigma_min = sigma_min #set to one to achieve same parameterization as for the BBED
        self.sigma_max = sigma_max #set to k to achieve same parameterization as for the BBED
        self.logsig = np.log(self.sigma_max / self.sigma_min)
        self.theta = theta

        self.ratio = self.sigma_max/self.sigma_min
        self.N = N
        self.Eilog = sc.expi(-2*self.logsig)
        self.T = T_sampling #for sampling in train step and inference
        self.Tc = 1 #for constructing the SDE



    def copy(self):
        return BBVE(self.T, self.sigma_min, self.sigma_max, self.theta, N=self.N)


    def T(self):
        return self.T
    
    def Tc(self):
        return self.Tc


    def sde(self, x, t, y):
        drift = (y - x)/(self.Tc - t)
        
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * np.sqrt(self.theta)
        
        return drift, diffusion


    def _mean(self, x0, t, y):
        time = (t/self.Tc)[:, None, None, None]
        mean = x0*(1-time) + y*time
        return mean

    def _std(self, t):
        t_np = t.cpu().detach().numpy()
        Eis = sc.expi(2*(t_np-1)*self.logsig) - self.Eilog
        k = 2*self.sigma_max**2*self.logsig
        var = self.sigma_min**2*(self.ratio**(2*t_np)-1+t_np) + k*(1-t_np)*Eis
        var = torch.tensor(var).to(device=t.device)*(1-t)*self.theta
        
        return torch.sqrt(var)

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(self.T*torch.ones((y.shape[0],), device=y.device))
        z = torch.randn_like(y)
        x_T = y + z * std[:, None, None, None]
        return x_T, z

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for BBVE not yet implemented!")

 """
