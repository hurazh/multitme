import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam

from multitme.utils import MLP, Exp, DatasetFromArray, GLASBEY_30_COLORS
import os
from tqdm.notebook import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class SSVAE(nn.Module):
    def __init__(
        self,
        ST_celltypes,
        data_path,
        output_size=10,
        input_size=784,
        z_dim=50,
        hidden_layers=(500,),
        config_enum=None,
        use_cuda=False,
        aux_loss_multiplier=100
    ):
        super().__init__()

        # initialize the class with all arguments provided to the constructor
        self.output_size = output_size
        self.input_size = input_size
        self.z_dim = z_dim
        self.hidden_layers = hidden_layers
        self.allow_broadcast = config_enum == "parallel"
        self.use_cuda = use_cuda
        self.aux_loss_multiplier = aux_loss_multiplier

        self.ST_celltypes = ST_celltypes
        self.IMC_biomarkers = np.load(os.path.join(data_path, 'imc_meta/IMC_biomarkers.npy'), allow_pickle=True)
        self.data_path = data_path
        self.setup_networks()
        logger.info('SSVAE initialized with output_size=%d, input_size=%d, z_dim=%d, hidden_layers=%s, aux_loss_multiplier=%d',
                    output_size, input_size, z_dim, hidden_layers, aux_loss_multiplier)


    def setup_networks(self):
        z_dim = self.z_dim
        hidden_sizes = self.hidden_layers

        self.encoder_y = MLP(
            [self.input_size] + hidden_sizes + [self.output_size],
            activation=nn.Softplus,
            output_activation=nn.Softmax,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )

        self.encoder_z = MLP(
            [self.input_size + self.output_size] + hidden_sizes + [[z_dim, z_dim]],
            activation=nn.Softplus,
            output_activation=[None, Exp],
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )

        self.decoder = MLP(
            [z_dim + self.output_size] + hidden_sizes + [[self.input_size, self.input_size]],
            activation=nn.Softplus,
            output_activation=[None, Exp],
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )

        # using GPUs for faster training of the networks
        if self.use_cuda:
            self.cuda()

    def model(self, xs, ys=None, ref=None):
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("ss_vae", self)

        batch_size = xs.size(0)
        options = dict(dtype=xs.dtype, device=xs.device)
        with pyro.plate("data", xs.shape[0]):
            # sample the handwriting style from the constant prior distribution
            prior_loc = torch.zeros(batch_size, self.z_dim, **options)
            prior_scale = torch.ones(batch_size, self.z_dim, **options)
            zs = pyro.sample("z", dist.Normal(prior_loc, prior_scale).to_event(1))

            # if the label y (which digit to write) is supervised, sample from the
            # constant prior, otherwise, observe the value (i.e. score it against the constant prior)
            alpha_prior = torch.ones(batch_size, self.output_size, **options) / (
                1.0 * self.output_size
            )
            ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=ys)
            
            x_loc, x_scale = self.decoder([zs, ys])
            pyro.sample(
                "x", dist.Normal(x_loc, x_scale).to_event(1), obs=xs
            )
            # return the loc so we can visualize it later
            return x_loc

    def guide(self, xs, ys=None, ref=None):
        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate("data", xs.shape[0]):
            alpha = self.encoder_y(xs)
            pyro.factor('spatial', (-ref*torch.log(alpha)).sum(), has_rsample=True)
#             pyro.factor('spatial', torch.abs(alpha - ref).sum(), has_rsample=True)
            if ys is None:
                ys = pyro.sample("y", dist.OneHotCategorical(alpha))

            # sample (and score) the latent var with the variational
            # distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
            loc, scale = self.encoder_z([xs, ys])
            pyro.sample("z", dist.Normal(loc, scale).to_event(1))

    def model_classify(self, xs, ys=None, ref=None):
        # register all pytorch (sub)modules with pyro
        pyro.module("ss_vae", self)

        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate("data", xs.shape[0]):
#             alpha = self.encoder_y(xs)
            if ys is not None:
                alpha = self.encoder_y(xs)
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    pyro.sample("y_aux", dist.OneHotCategorical(alpha), obs=ys)

    def guide_classify(self, xs, ys=None, ref=None):
        pass

    def train(self, losses, train_sup_loader, train_unsup_loader):
        num_losses = len(losses)

        # compute number of batches for an epoch
        sup_batches = len(train_sup_loader)
        unsup_batches = len(train_unsup_loader)
        batches_per_epoch = sup_batches + unsup_batches
        periodic_interval_batches = batches_per_epoch // sup_batches
        # initialize variables to store loss values
        epoch_losses_sup = [0.0] * num_losses
        epoch_losses_unsup = [0.0] * num_losses

        # setup the iterators for training data loaders
        sup_iter = iter(train_sup_loader)
        unsup_iter = iter(train_unsup_loader)

        # count the number of supervised batches seen in this epoch
        ctr_sup = 0
        for i in range(batches_per_epoch):
            # whether this batch is supervised or not
            is_supervised = (i % periodic_interval_batches == 1) and ctr_sup < sup_batches

            # extract the corresponding batch
            if is_supervised:
                (xs, ys, ref) = next(sup_iter)
                ctr_sup += 1
            else:
                (xs, ys, ref) = next(unsup_iter)

            # run the inference for each loss with supervised or un-supervised
            # data as arguments
            for loss_id in range(num_losses):
                if is_supervised:
                    epoch_losses_sup[loss_id] += losses[loss_id].step(xs, torch.nn.functional.one_hot(ys, num_classes=22), ref=ref)
                else:
                    epoch_losses_unsup[loss_id] += losses[loss_id].step(xs, ref=ref)
        return epoch_losses_sup, epoch_losses_unsup

    def predict(self):
        z_val = np.load(os.path.join(self.data_path, 'imc_reads/Obs_imc.npy'))
        z_log = (z_val - z_val.mean(axis=0))/(np.std(z_val, axis=0))
        alpha = self.encoder_y(torch.tensor(z_log, dtype=torch.float32)).detach()
        alpha[alpha < 0.05] = 0
        alpha = alpha / alpha.sum(axis=1, keepdims=True)
        pred_dist = dist.OneHotCategorical(alpha)
        pred_celltype = pred_dist.sample()
        return pred_celltype

    def fit(self, cell_types, biomarkers, batch_size=100, num_epochs=200, learning_rate=0.001, test_frequency=5, betas=(0.9, 0.999)):
        train_sup_loader, train_unsup_loader, test_sup_loader = self.create_data_loaders(cell_types, biomarkers, batch_size=batch_size)

        pyro.clear_param_store()
        
        adam_params = {"lr": learning_rate, "betas": betas}
        optimizer = Adam(adam_params)
        losses = []

        elbo = Trace_ELBO()
        loss_base = SVI(self.model, self.guide, optimizer, loss=elbo)
        losses.append(loss_base)

        # aux_loss: whether to use the auxiliary loss from NIPS 14 paper (Kingma et al.)
        loss_aux = SVI(self.model_classify, self.guide_classify, optimizer, loss=elbo)
        losses.append(loss_aux)

        train_loss_sup = []
        train_loss_unsup = []
        
        with tqdm(range(0, num_epochs), unit="epoch") as tepoch:
            tepoch.set_description('Train')
            for epoch in tepoch:
                epoch_losses_sup, epoch_losses_unsup = self.train(losses, train_sup_loader, train_unsup_loader)
                train_loss_sup.append(epoch_losses_sup)
                train_loss_unsup.append(epoch_losses_unsup)
                tepoch.set_postfix(L_rs=epoch_losses_sup[0], 
                                   L_cs=epoch_losses_sup[1],
                                   L_ru=epoch_losses_unsup[0])
                
                if (epoch + 1) % test_frequency == 0:
                    pass
                
        return train_loss_sup, train_loss_unsup


    def create_data_loaders(self, cell_types, biomarkers, batch_size=100):
        z_val = np.load(os.path.join(self.data_path, 'imc_reads/Obs_imc.npy'))
        z_log = (z_val - z_val.mean(axis=0))/(np.std(z_val, axis=0))
        spatial_ref = np.load(os.path.join(self.data_path, 'imc_reads/ST_spatial_reference.npy'))
        # Find corresponding indices for each cell type
        selected_idx = []
        selected_labels = []
        
        for i, (cell_type, biomarker) in enumerate(zip(cell_types, biomarkers)):
            st_idx = np.where(self.ST_celltypes == cell_type)[0][0]
            imc_idx = np.where(self.IMC_biomarkers == biomarker)[0][0]
            cell_idx = z_log[:, imc_idx].argsort()[::-1][:100]
            cell_labels = np.ones_like(cell_idx) * st_idx
            selected_idx.append(cell_idx)
            selected_labels.append(cell_labels)
        
        selected_idx = np.concatenate(selected_idx)
        selected_labels = np.concatenate(selected_labels)
        
        shuffle_idx = np.arange(len(selected_idx))
        np.random.shuffle(shuffle_idx)
        
        train_sup_idx = selected_idx[shuffle_idx[100:]]
        test_sup_idx = selected_idx[shuffle_idx[:100]]
        train_sup_labels = selected_labels[shuffle_idx[100:]]
        test_sup_labels = selected_labels[shuffle_idx[:100]]
        
        torch.manual_seed(0)
        
        unsupervised_dataset = DatasetFromArray(torch.tensor(z_log, dtype=torch.float32), 
                                                torch.tensor(np.zeros_like(z_log)),
                                                torch.tensor(spatial_ref, dtype=torch.float32))
        supervised_dataset = DatasetFromArray(torch.tensor(z_log[train_sup_idx], dtype=torch.float32), 
                                              torch.tensor(train_sup_labels),
                                              torch.tensor(spatial_ref[train_sup_idx], dtype=torch.float32))
        supervised_testset = DatasetFromArray(torch.tensor(z_log[test_sup_idx], dtype=torch.float32), 
                                              torch.tensor(test_sup_labels),
                                              torch.tensor(spatial_ref[test_sup_idx], dtype=torch.float32))

        train_unsup_loader = DataLoader(dataset=unsupervised_dataset, batch_size=batch_size, shuffle=True)
        train_sup_loader = DataLoader(dataset=supervised_dataset, batch_size=batch_size, shuffle=True)
        test_sup_loader = DataLoader(dataset=supervised_testset, batch_size=batch_size, shuffle=True)
        
        return train_sup_loader, train_unsup_loader, test_sup_loader

    def plot_IMConST(self, seed=0, legend=False):
        torch.manual_seed(seed)
        pred_celltype = self.predict()
        from PIL import Image
        ST_img = Image.open(os.path.join(self.data_path, 'ST_img.jpg'))
        all_imc_cells = np.load(os.path.join(self.data_path, 'imc_meta/all_imc_cells.npy'))
        ST_img = np.array(ST_img)[::-1]
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.imshow(ST_img)
        for i in range(22):
            ct_idx = pred_celltype[:, i] > 0
            ax.scatter(all_imc_cells[ct_idx, 0], all_imc_cells[ct_idx, 1], c=GLASBEY_30_COLORS[i], s=5, edgecolors='none',
                      label=self.ST_celltypes[i])
        ax.set_axis_off()
        if legend:
            plt.legend()
        if not os.path.exists(os.path.join(self.data_path, 'plots')):
                os.makedirs(os.path.join(self.data_path, 'plots'))
        plt.savefig(os.path.join(self.data_path, 'plots/IMC_on_ST.pdf'), bbox_inches='tight')