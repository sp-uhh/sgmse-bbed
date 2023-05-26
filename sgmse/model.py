import time
from math import ceil
import warnings
import numpy as np
import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import torch.nn.functional as F
from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model
from sgmse.util.other import pad_spec
import numpy as np
import matplotlib.pyplot as plt



class ScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--num_eval_files", type=int, default=10, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="mse", help="The type of loss function to use.")
        parser.add_argument("--loss_abs_exponent", type=float, default= 0.5,  help="magnitude transformation in the loss term")
        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.999, t_eps=3e-2, loss_abs_exponent=0.5, 
        num_eval_files=10, loss_type='mse', data_module_cls=None, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
        # Initialize SDE
        if sde == 'bbve':
            #change parameters, if the old class bbve is used. Needed for loading the provided checkpoint
            #as that checkpoint was trained with the old class.
            sde = 'bbed'
            kwargs['k'] = kwargs['sigma_max']
            del kwargs['sigma_max']
            del kwargs['sigma_min']
        
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files
        self.loss_abs_exponent = loss_abs_exponent
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    
    def _loss(self, score, sigmas, z):    
        if self.loss_type == 'mse':
            err = sigmas*score + z 
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def _step(self, batch, batch_idx):
        x, y = batch
        rdm = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        t = torch.min(rdm, torch.tensor(self.sde.T))
        mean, std = self.sde.marginal_prob(x, t, y)
        z = torch.randn_like(x)  #
        sigmas = std[:, None, None, None]
        perturbed_data = mean + sigmas * z
        score = self(perturbed_data, t, y)
        loss = self._loss(score, sigmas, z)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files)
            self.log('pesq', pesq, on_step=False, on_epoch=True)
            self.log('si_sdr', si_sdr, on_step=False, on_epoch=True)
            self.log('estoi', estoi, on_step=False, on_epoch=True)

        return loss

    def forward(self, x, t, y):
        # Concatenate y as an extra channel
        dnn_input = torch.cat([x, y], dim=1)
        
        # the minus is most likely unimportant here - taken from Song's repo
        score = -self.dnn(dnn_input, t)
        return score

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, Y_prior=None, N=None, minibatch=None, timestep_type=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, Y=y, Y_prior=Y_prior,
             timestep_type=timestep_type, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    y_prior_mini = Y_prior[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, Y=y_mini, y_prior=y_prior_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y,  Y_prior=None, N=None, minibatch=None, timestep_type=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, Y_prior=Y_prior,
             timestep_type=timestep_type, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns
            return batched_sampling_fn

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)



    def enhance_debug(self, y, x=None, sampler_type="pc", predictor="reverse_diffusion",
        corrector="ald", N=30, corrector_steps=1, snr=0.5, timeit=False, timestep_type=None,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        sr=16000
        start = time.time()
        T_orig = y.size(1) 
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        x = x / norm_factor
        
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        
        
        X = torch.unsqueeze(self._forward_transform(self._stft(x.cuda())), 0)
        X = pad_spec(X)
               
        if len(Y.shape)==4:
            Y = Y*self.preemp[None, None, :, None].to(device=Y.device)
            X = X*self.preemp[None, None, :, None].to(device=X.device)
        elif len(Y.self.shape)==3:
            Y = Y*self.preemp[None, :, None].to(device=Y.device)
            X = X*self.preemp[None, :, None].to(device=X.device)
        else:
            Y = Y*self.preemp[:, None].to(device=Y.device)
            X = X*self.preemp[:, None].to(device=X.device)
        
        Y_prior = self.sde._mean(X, torch.tensor([self.sde.T]).cuda(), Y)
        if sampler_type == "pc":
            sampler = self.get_pc_sampler(predictor, corrector, Y.cuda(), Y_prior = Y_prior.cuda(), N=N, 
                corrector_steps=corrector_steps, snr=snr, intermediate=False, timestep_type=timestep_type,
                **kwargs)
        elif sampler_type == "ode":
            sampler = self.get_ode_sampler(Y.cuda(), N=N, **kwargs)
        else:
            print("{} is not a valid sampler type!".format(sampler_type))
        sample, nfe = sampler()
        
        
        sample = sample.squeeze()
        if len(sample.shape)==4:
            sample = sample*self.deemp[None, None, :, None].to(device=sample.device)
        elif len(sample.shape)==3:
            sample = sample*self.deemp[None, :, None].to(device=sample.device)
        else:
            sample = sample*self.deemp[:, None].to(device=sample.device)
        
        x_hat = self.to_audio(sample, T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        if timeit:
            rtf = (end-start)/(len(x_hat)/sr)
            return x_hat, nfe, rtf
        else:
            return x_hat



    def enhance(self, y, sampler_type="pc", predictor="reverse_diffusion",
        corrector="ald", N=30, corrector_steps=1, snr=0.5, timeit=False,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        sr=16000
        start = time.time()
        T_orig = y.size(1) 
        norm_factor = y.abs().max().item()
        y = y / norm_factor

        
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        
            
        
        
        if sampler_type == "pc":
            sampler = self.get_pc_sampler(predictor, corrector, Y.cuda(), N=N, 
                corrector_steps=corrector_steps, snr=snr, intermediate=False,
                **kwargs)
        elif sampler_type == "ode":
            sampler = self.get_ode_sampler(Y.cuda(), N=N, **kwargs)
        else:
            print("{} is not a valid sampler type!".format(sampler_type))
        sample, nfe = sampler()
        
        
        sample = sample.squeeze()

        
        x_hat = self.to_audio(sample, T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        if timeit:
            rtf = (end-start)/(len(x_hat)/sr)
            return x_hat, nfe, rtf
        else:
            return x_hat


    def prior_tests2(self, y, x, n):
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        x = x / norm_factor
        n = n / norm_factor
        
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        
        X = torch.unsqueeze(self._forward_transform(self._stft(x.cuda())), 0)
        X = pad_spec(X)
        
        #Ns = torch.unsqueeze(self._forward_transform(self._stft(n.cuda())), 0)
        #Ns = pad_spec(Ns)
        Ns = Y-X
        
        
        if len(Y.shape)==4:
            Y = Y*self.preemp[None, None, :, None].to(device=Y.device)
            Ns = Ns*self.preemp[None, None, :, None].to(device=Y.device)
            X = X*self.preemp[None, None, :, None].to(device=Y.device)
        elif len(Y.self.shape)==3:
            Y = Y*self.preemp[None, :, None].to(device=Y.device)
        else:
            Y = Y*self.preemp[:, None].to(device=Y.device)
            X = X*self.preemp[:, None].to(device=X.device)
            Ns = Ns*self.preemp[:, None].to(device=Ns.device)
        
        
        Yt, z = self.sde.prior_sampling(Y.shape, Y)
        Yt = Yt.to(Y.device)
        z = z.to(Y.device)
        
        vec_t = torch.ones(Y.shape[0], device=Y.device) * torch.tensor([1.0], device=Y.device)
        
        with torch.no_grad():
            
            grad = self(Yt, vec_t, Y)
            std = self.sde._std(vec_t)

            mp = Yt + grad*(std**2)
            mp_np = mp.squeeze().detach().cpu().numpy()
            
            z = z #/std
            z_np = z.squeeze().detach().cpu().numpy()
            
            Y_np = Y.squeeze().detach().cpu().numpy()
            Ns_np = Ns.squeeze().detach().cpu().numpy()
            X_np = X.squeeze().detach().cpu().numpy()
            
            Yt_np = Yt.squeeze().detach().cpu().numpy()
            grad_np = grad.squeeze().detach().cpu().numpy()
            
            res = z_np+grad_np
            err = np.exp(-1.5)*res/np.max(np.abs(res)) - np.exp(-1.5)*Ns_np/np.max(np.abs(Ns_np))
            #mean_res = (mp_np - Y_np) + 1e-8
            #err = (Ns_np)/(mean_res + 1e-8)
            
            fig, axs = plt.subplots(3, 3, figsize=(10,9), sharex=True, sharey=True)
            
            axs[1,0].imshow(20*np.log10(np.abs(grad_np)), aspect='auto', vmin=-30, vmax=30, origin='lower', cmap='magma')
            axs[1,0].set_title('predicted score')
            axs[1,0].set_xlabel('time [s]')
            axs[1,0].set_ylabel('frequency [kHz]')
            
            axs[1,1].imshow(20*np.log10(np.abs(Yt_np)), aspect='auto', vmin=-30, vmax=30, origin='lower', cmap='magma')
            axs[1,1].set_title('yT = y + z*sigma(T)')
            axs[1,1].set_xlabel('time [s]')
            axs[1,1].set_ylabel('frequency [kHz]')
            
            im = axs[1,2].imshow(20*np.log10(np.abs(mp_np)), aspect='auto', vmin=-30, vmax=30, origin='lower', cmap='magma')
            axs[1,2].set_title('mean = yT + score*sigma(T)^2')
            axs[1,2].set_xlabel('time [s]')
            axs[1,2].set_ylabel('frequency [kHz]')
            
            
            im = axs[2,0].imshow(20*np.log10(np.abs(res/np.max(np.abs(res)))), aspect='auto', vmin=-30, vmax=30, origin='lower', cmap='magma')
            axs[2,0].set_title('score + z/sigma(T)')
            axs[2,0].set_xlabel('time [s]')
            axs[2,0].set_ylabel('frequency [kHz]')
            
            
            im = axs[0,2].imshow(20*np.log10(np.abs(Y_np)), aspect='auto', vmin=-30, vmax=30, origin='lower', cmap='magma')
            axs[0,2].set_title('noisy mixture')
            axs[0,2].set_xlabel('time [s]')
            axs[0,2].set_ylabel('frequency [kHz]')
            
            
            im = axs[2,1].imshow(20*np.log10(np.abs(mp_np - Y_np)), aspect='auto', vmin=-30, vmax=30, origin='lower', cmap='magma')
            axs[2,1].set_title('recon mean - noisy mixture')
            axs[2,1].set_xlabel('time [s]')
            axs[2,1].set_ylabel('frequency [kHz]')

            
            axs[0,0].imshow(20*np.log10(np.abs(X_np)), aspect='auto', vmin=-30, vmax=30, origin='lower', cmap='magma')
            axs[0,0].set_title('Clean')
            axs[0,0].set_xlabel('time [s]')
            axs[0,0].set_ylabel('frequency [kHz]')
            
            axs[0,1].imshow(20*np.log10(np.abs(Ns_np/np.max(np.abs(Ns_np)))), aspect='auto', vmin=-30, vmax=30, origin='lower', cmap='magma')
            axs[0,1].set_title('environmental noise')
            axs[0,1].set_xlabel('time [s]')
            axs[0,1].set_ylabel('frequency [kHz]')
            
            
            axs[2,2].imshow(20*np.log10(np.abs(err)), aspect='auto', vmin=-30, vmax=30, origin='lower', cmap='magma')
            axs[2,2].set_title('err')
            axs[2,2].set_xlabel('time [s]')
            axs[2,2].set_ylabel('frequency [kHz]')
            
            fig.tight_layout()
            fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
            plt.show()
            plt.savefig('blub.png')
            a=2


    def get_prior(self, y, x, n, T=1):
            norm_factor = y.abs().max().item()
            y = y / norm_factor
            x = x / norm_factor
            n = n / norm_factor

            Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
            Y = pad_spec(Y)
            
            X = torch.unsqueeze(self._forward_transform(self._stft(x.cuda())), 0)
            diff_pad = Y.shape[-1] - X.shape[-1]
            X = pad_spec(X)
            
            Ns = Y - X

            if len(Y.shape)==4:
                Y = Y*self.preemp[None, None, :, None].to(device=Y.device)
                Ns = Ns*self.preemp[None, None, :, None].to(device=Y.device)
                X = X*self.preemp[None, None, :, None].to(device=Y.device)
            elif len(Y.self.shape)==3:
                Y = Y*self.preemp[None, :, None].to(device=Y.device)
            else:
                Y = Y*self.preemp[:, None].to(device=Y.device)
                X = X*self.preemp[:, None].to(device=X.device)
                Ns = Ns*self.preemp[:, None].to(device=Ns.device)
            
            if self.sde.__class__.__name__ == 'BBVE':
                self.sde.T = T
            Yt, z = self.sde.prior_sampling(Y.shape, Y)
            Yt = Yt.to(Y.device)
            z = z.to(Y.device)
            
            vec_t = torch.ones(Y.shape[0], device=Y.device) * torch.tensor([T], device=Y.device)
   
            grad = self(Yt, vec_t, Y, vec_t[:, None, None, None])
            std = self.sde._std(vec_t)

            mp = Yt + grad*(std**2)
            mp_np = mp.squeeze().detach().cpu().numpy()
            
            z = z/std
            z_np = z.squeeze().detach().cpu().numpy()
            
            Y_np = Y.squeeze().detach().cpu().numpy()
            X_np = X.squeeze().detach().cpu().numpy()
            Ns_np = Ns.squeeze().detach().cpu().numpy()
            
            Yt_np = Yt.squeeze().detach().cpu().numpy()
            grad_np = grad.squeeze().detach().cpu().numpy()
            
            res = z_np+grad_np

            return mp_np[:, :-diff_pad], X_np[:, :-diff_pad], Y_np[:, :-diff_pad], res[:, :-diff_pad], z_np[:, :-diff_pad], grad_np[:, :-diff_pad], Ns_np[:, :-diff_pad]
        

                
             
        
