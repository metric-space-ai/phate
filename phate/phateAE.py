import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return F.relu(self.linear(x))


class MLP(nn.Sequential):
    def __init__(self, dim_list, sigmoid=False):
        modules = [LinearBlock(dim_list[i - 1], dim_list[i]) for i in range(1, len(dim_list) - 1)]
        modules.append(nn.Linear(dim_list[-2], dim_list[-1]))

        if sigmoid:
            modules.append(nn.Sigmoid())

        super().__init__(*modules)


class AutoencoderModule(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim, noise=0, vae=False, sigmoid=False):
        super().__init__()
        self.vae = vae

        full_list = [input_dim] + list(hidden_dims) + [z_dim * 2 if vae else z_dim]
        self.encoder = MLP(dim_list=full_list)

        full_list.reverse()
        full_list[0] = z_dim
        self.decoder = MLP(dim_list=full_list, sigmoid=sigmoid)
        self.noise = noise

    def forward(self, x):
        z = self.encoder(x)

        if self.noise > 0:
            z_decoder = z + self.noise * torch.randn_like(z)
        else:
            z_decoder = z

        if self.vae:
            mu, logvar = z.chunk(2, dim=-1)

            if self.training:
                z_decoder = mu + torch.exp(logvar / 2.) * torch.randn_like(logvar)
            else:
                z_decoder = mu

        output = self.decoder(z_decoder)
        return output, z


class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.max = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.max(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = F.relu(x)
        return x


class LastConv(UpConvBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, in_channels)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = super().forward(x)
        x = self.conv_2(x)
        return x


class ConvEncoder(nn.Module):
    def __init__(self, H, W, input_channel, channel_list, hidden_dims, z_dim):
        super().__init__()

        channels = [input_channel] + channel_list
        modules = [DownConvBlock(channels[i - 1], channels[i]) for i in range(1, len(channels))]
        self.conv = nn.Sequential(*modules)

        factor = 2 ** len(channel_list)
        self.fc_size = int(channel_list[-1] * H / factor * W / factor)
        mlp_dim = [self.fc_size] + hidden_dims + [z_dim]
        self.linear = MLP(mlp_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.fc_size)
        x = self.linear(x)
        return x


class ConvDecoder(nn.Module):
    def __init__(self, H, W, input_channel, channel_list, hidden_dims, z_dim, sigmoid):
        super().__init__()
        self.H = H
        self.W = W
        self.factor = 2 ** len(channel_list)
        fc_size = int(channel_list[0] * H / self.factor * W / self.factor)
        mlp_dim = [z_dim] + hidden_dims + [fc_size]
        self.linear = MLP(mlp_dim)

        channels = channel_list
        modules = [UpConvBlock(channels[i - 1], channels[i]) for i in range(1, len(channels))]
        modules.append(LastConv(channels[-1], input_channel))
        if sigmoid:
            modules.append(nn.Sigmoid())
        self.conv = nn.Sequential(*modules)
        self.first_channel = channel_list[0]

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.first_channel, self.H // self.factor, self.W // self.factor)
        x = self.conv(x)
        return x


class ConvAutoencoderModule(nn.Module):
    def __init__(self, H, W, input_channel, channel_list, hidden_dims, z_dim, noise, vae=False, sigmoid=False):
        super().__init__()
        self.vae = vae

        self.encoder = ConvEncoder(H, W, input_channel, channel_list, hidden_dims, z_dim * 2 if self.vae else z_dim)
        channel_list.reverse()
        hidden_dims.reverse()
        self.decoder = ConvDecoder(H, W, input_channel, channel_list, hidden_dims, z_dim, sigmoid)
        self.noise = noise

    def forward(self, x):
        return AutoencoderModule.forward(self, x)

import time

import matplotlib
import numpy as np
from sklearn.metrics import mean_squared_error

SEED = 42

class BaseModel:
    #All models should subclass BaseModel.
    def __init__(self):
        """Init."""
        self.comet_exp = None

    def fit(self, x):
        """Fit model to data.
        Args:
            x(BaseDataset): Dataset to fit.
        """
        raise NotImplementedError()

    def fit_transform(self, x):
        """Fit model and transform data.
        If model is a dimensionality reduction method, such as an Autoencoder, this should return the embedding of X.
        Args:
            x(BaseDataset): Dataset to fit and transform.
        Returns:
            ndarray: Embedding of x.
        """
        self.fit(x)
        return self.transform(x)

    def transform(self, x):
        """Transform data.
        If model is a dimensionality reduction method, such as an Autoencoder, this should return the embedding of x.
        Args:
            X(BaseDataset): Dataset to transform.
        Returns:
            ndarray: Embedding of X.
        """
        raise NotImplementedError()

    def inverse_transform(self, x):
        """Take coordinates in the embedding space and invert them to the data space.
        Args:
            x(ndarray): Points in the embedded space with samples on the first axis.
        Returns:
            ndarray: Inverse (reconstruction) of x.
        """
        raise NotImplementedError()

import numpy as np
import phate
from sklearn.decomposition import PCA as SKPCA
from sklearn.pipeline import make_pipeline
from tqdm import tqdm


def procrustes(X, Y, scaling=True, reflection='best'):
    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2).sum()
    ssY = (Y0**2).sum()

    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros((n, m - my))), 0)

    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':
        have_reflection = np.linalg.det(T) < 0
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform


PROC_THRESHOLD = 20000
PROC_BATCH_SIZE = 5000
PROC_LM = 1000


def fit_transform_procrustes(x, fit_transform_call, procrustes_batch_size, procrustes_lm):
    lm_points = x[:procrustes_lm, :]  # Reference points included in all batches
    initial_embedding = fit_transform_call(lm_points)
    result = [initial_embedding]
    remaining_x = x[procrustes_lm:, :]

    n_batches = (len(remaining_x) + procrustes_batch_size - 1) // procrustes_batch_size
    for i in tqdm(range(n_batches), desc='Processing Batches'):
        start_idx = i * procrustes_batch_size
        end_idx = min((i + 1) * procrustes_batch_size, len(remaining_x))

        new_points = remaining_x[start_idx:end_idx, :]
        subsetx = np.vstack((lm_points, new_points))
        subset_embedding = fit_transform_call(subsetx)

        d, Z, tform = procrustes(initial_embedding,
                                 subset_embedding[:procrustes_lm, :])

        subset_embedding_transformed = np.dot(
            subset_embedding[procrustes_lm:, :],
            tform['rotation']) + tform['translation']

        result.append(subset_embedding_transformed)

    return np.vstack(result)


class PHATE(phate.PHATE):
    def __init__(self, proc_threshold=PROC_THRESHOLD, procrustes_batches_size=PROC_BATCH_SIZE,
                 procrustes_lm=PROC_LM, **kwargs):
        self.proc_threshold = proc_threshold
        self.procrustes_batch_size = procrustes_batches_size
        self.procrustes_lm = procrustes_lm
        self.comet_exp = None
        super().__init__(**kwargs)

    def fit_transform(self, x):
        x, _ = x.numpy()

        if x.shape[0] < self.proc_threshold:
            result = super().fit_transform(x)
        else:
            print('Fitting procrustes...')
            result = self.fit_transform_procrustes(x)
        return result

    def fit_transform_procrustes(self, x):
        return fit_transform_procrustes(x, super().fit_transform, self.procrustes_batch_size, self.procrustes_lm)
    

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FIT_DEFAULT = .85  # Default train/test split ratio
SEED = 42  # Default seed for splitting

DEFAULT_PATH = os.path.join(os.getcwd(), 'data')


class FromNumpyDataset(Dataset):
    """Torch Dataset Wrapper for x ndarray with no target."""

    def __init__(self, x):
        """Initialize torch wrapper dataset from simple ndarray.

        Args:
            x (ndarray): Input variables.
        """
        self._data = torch.from_numpy(x).float()

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def numpy(self, idx=None):
        """Get dataset as ndarray.

        Specify indices to return a subset of the dataset, otherwise return whole dataset.

        Args:
            idx(int, optional): Specify index or indices to return.

        Returns:
            ndarray: Return flattened dataset as a ndarray.

        """
        n = len(self)

        data = self._data.numpy().reshape((n, -1))

        if idx is None:
            return data
        else:
            return data[idx]


class BaseDataset(Dataset):
    """Template class for all datasets in the project.

    All datasets should subclass BaseDataset, which contains built-in splitting utilities."""

    def __init__(self, x, y, split, split_ratio, random_state, labels=None):
        """Initialize dataset.

        Set the split parameter to 'train' or 'test' for the object to hold the desired split. split='none' will keep
        the entire dataset in the attributes.

        Args:
            x(ndarray): Input features.
            y(ndarray): Targets.
            split(str): Name of split.
            split_ratio(float): Ratio to use for train split. Test split ratio is 1 - split_ratio.
            random_state(int): To set random_state values for reproducibility.
            labels(ndarray, optional): Specify labels for stratified splits.
        """
        if split not in ('train', 'test', 'none'):
            raise ValueError('split argument should be "train", "test" or "none"')

        # Get train or test split
        x, y = self.get_split(x, y, split, split_ratio, random_state, labels)

        self.data = x.float()
        self.targets = y.float()  # One target variable. Used mainly for coloring.
        self.latents = None  # Arbitrary number of continuous ground truth variables. Used for computing metrics.

        # Arbitrary number of label ground truth variables. Used for computing metrics.
        # Should range from 0 to no_of_classes -1
        self.labels = None
        self.is_radial = []  # Indices of latent variable requiring polar conversion when probing (e.g. Teapot, RotatedDigits)
        self.partition = True  # If labels should be used to partition the data before regressing latent factors. See score.EmbeddingProber.

    def __getitem__(self, index):
        return self.data[index], self.targets[index], index

    def __len__(self):
        return len(self.data)

    def numpy(self, idx=None):
        """Get dataset as ndarray.

        Specify indices to return a subset of the dataset, otherwise return whole dataset.

        Args:
            idx(int, optional): Specify index or indices to return.

        Returns:
            ndarray: Return flattened dataset as a ndarray.

        """
        n = len(self)

        data = self.data.numpy().reshape((n, -1))

        if idx is None:
            return data, self.targets.numpy()
        else:
            return data[idx], self.targets[idx].numpy()

    def get_split(self, x, y, split, split_ratio, random_state, labels=None):
        """Split dataset.

        Args:
            x(ndarray): Input features.
            y(ndarray): Targets.
            split(str): Name of split.
            split_ratio(float): Ratio to use for train split. Test split ratio is 1 - split_ratio.
            random_state(int): To set random_state values for reproducibility.
            labels(ndarray, optional): Specify labels for stratified splits.

        Returns:
            (tuple): tuple containing :
                    x(ndarray): Input variables in requested split.
                    y(ndarray): Target variable in requested split.
        """
        if split == 'none':
            return torch.from_numpy(x), torch.from_numpy(y)

        n = x.shape[0]
        train_idx, test_idx = train_test_split(np.arange(n),
                                               train_size=split_ratio,
                                               random_state=random_state,
                                               stratify=labels)

        if split == 'train':
            return torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])
        else:
            return torch.from_numpy(x[test_idx]), torch.from_numpy(y[test_idx])

    def get_latents(self):
        """Get latent variables.

        Returns:
            latents(ndarray): Latent variables for each sample.
        """
        return self.latents

    def random_subset(self, n, random_state):
        """Get random subset of dataset.

        Args:
            n(int): Number of samples to subset.
            random_state(int): Seed for reproducibility

        Returns:
            Subset(TorchDataset) : Random subset.

        """

        np.random.seed(random_state)
        sample_mask = np.random.choice(len(self), n, replace=False)

        next_latents = self.latents[sample_mask] if self.latents is not None else None
        next_labels = self.labels[sample_mask] if self.labels is not None else None

        return NoSplitBaseDataset(self.data[sample_mask], self.targets[sample_mask], next_latents, next_labels)

    def validation_split(self, ratio=.15 / FIT_DEFAULT, random_state=42):
        """Randomly subsample validation split in self.

        Return both train split and validation split as two different BaseDataset objects.

        Args:
            ratio(float): Ratio of train split to allocate to validation split. Default option is to sample 15 % of
            full dataset, by adjusting with the initial train/test ratio.
            random_state(int): Seed for sampling.

        Returns:
            (tuple) tuple containing:
                x_train(BaseDataset): Train set.
                x_val(BaseDataset): Val set.

        """

        np.random.seed(random_state)
        sample_mask = np.random.choice(len(self), int(ratio * len(self)), replace=False)
        val_mask = np.full(len(self), False, dtype=bool)
        val_mask[sample_mask] = True
        train_mask = np.logical_not(val_mask)
        next_latents_train = self.latents[train_mask] if self.latents is not None else None
        next_latents_val = self.latents[val_mask] if self.latents is not None else None
        next_labels_train = self.labels[train_mask] if self.labels is not None else None
        next_labels_val = self.labels[val_mask] if self.labels is not None else None

        x_train = NoSplitBaseDataset(self.data[train_mask], self.targets[train_mask],
                                     next_latents_train, next_labels_train)
        x_val = NoSplitBaseDataset(self.data[val_mask], self.targets[val_mask],
                                   next_latents_val, next_labels_val)

        return x_train, x_val

class BaseDataset2(Dataset):
    """Template class for all datasets in the project."""

    def __init__(self, x, y):
        """Initialize dataset.

        Args:
            x(ndarray): Input features.
            y(ndarray): Targets.
        """
        self.data = torch.from_numpy(x).float()
        self.targets = torch.from_numpy(y).float()
        self.latents = None

        self.labels = None
        self.is_radial = []
        self.partition = True

    def __getitem__(self, index):
        return self.data[index], self.targets[index], index

    def __len__(self):
        return len(self.data)

    def numpy(self, idx=None):
        """Get dataset as ndarray.

        Specify indices to return a subset of the dataset, otherwise return whole dataset.

        Args:
            idx(int, optional): Specify index or indices to return.

        Returns:
            ndarray: Return flattened dataset as a ndarray.

        """
        n = len(self)

        data = self.data.numpy().reshape((n, -1))

        if idx is None:
            return data, self.targets.numpy()
        else:
            return data[idx], self.targets[idx].numpy()

    def get_latents(self):
        """Get latent variables.

        Returns:
            latents(ndarray): Latent variables for each sample.
        """
        return self.latents

class NoSplitBaseDataset(BaseDataset):
    #BaseDataset class when splitting is not required and x and y are already torch tensors.

    def __init__(self, x, y, latents, labels):
        """Initialize dataset.

        Args:
            x(ndarray): Input variables.
            y(ndarray): Target variable. Used for coloring.
            latents(ndarray): Other continuous target variable. Used for metrics.
            labels(ndarray): Other label target variable. Used for metrics.
        """
        self.data = x.float()
        self.targets = y.float()
        self.latents = latents
        self.labels = labels

import os
import torch
import torch.nn as nn
import numpy as np
import scipy

# Default hyperparameters
BATCH_SIZE = 128
LR = .0001
WEIGHT_DECAY = 0
EPOCHS = 200
HIDDEN_DIMS = (800, 400, 200)  # Default fully-connected dimensions
CONV_DIMS = [32, 64]  # Default conv channels
CONV_FC_DIMS = [400, 200]  # Default fully-connected dimensions after convs


class AE(BaseModel):
    """Vanilla Autoencoder model.

    Trained with Adam and MSE Loss.
    Model will infer from the data whether to use a fully FC or convolutional + FC architecture.
    """

    def __init__(self, *,
                 lr=LR,
                 epochs=EPOCHS,
                 batch_size=BATCH_SIZE,
                 weight_decay=WEIGHT_DECAY,
                 random_state=SEED,
                 n_components=2,
                 hidden_dims=HIDDEN_DIMS,
                 conv_dims=CONV_DIMS,
                 conv_fc_dims=CONV_FC_DIMS,
                 noise=0,
                 patience=50,
                 data_val=None,
                 comet_exp=None,
                 write_path=''):
        """Initialize the autoencoder.

        Args:
            lr(float): Learning rate.
            epochs(int): Number of epochs for model training.
            batch_size(int): Mini-batch size.
            weight_decay(float): L2 penalty.
            random_state(int): To seed parameters and training routine for reproducible results.
            n_components(int): Bottleneck dimension.
            hidden_dims(List[int]): Number and size of fully connected layers for encoder. Do not specify the input
            layer or the bottleneck layer, since they are inferred from the data or from the n_components
            argument respectively. Decoder will use the same dimensions in reverse order. This argument is only used if
            provided samples are flat vectors.
            conv_dims(List[int]): Specify the number of convolutional layers. The int values specify the number of
            channels for each layer. This argument is only used if provided samples are images (i.e. 3D tensors)
            conv_fc_dims(List[int]): Number and size of fully connected layers following the conv_dims convolutionnal
            layer. No need to specify the bottleneck layer. This argument is only used if provided samples
            are images (i.e. 3D tensors)
            noise(float): Variance of the gaussian noise injected in the bottleneck before reconstruction.
            patience(int): Epochs with no validation MSE improvement before early stopping.
            data_val(BaseDataset): Split to validate MSE on for early stopping.
            comet_exp(Experiment): Comet experiment to log results.
            write_path(str): Where to write temp files.
        """
        self.random_state = random_state
        self.n_components = n_components
        self.hidden_dims = hidden_dims
        self.fitted = False  # If model was fitted
        self.torch_module = None  # Will be initialized to the appropriate torch module when fit method is called
        self.optimizer = None  # Will be initialized to the appropriate optimizer when fit method is called
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.criterion = nn.MSELoss(reduction='mean')
        self.conv_dims = conv_dims
        self.conv_fc_dims = conv_fc_dims
        self.noise = noise
        self.comet_exp = comet_exp
        self.data_shape = None  # Shape of input data

        # Early stopping attributes
        self.data_val = data_val
        self.val_loader = None
        self.patience = patience
        self.current_loss_min = np.inf
        self.early_stopping_count = 0
        self.write_path = write_path

    def init_torch_module(self, data_shape, vae=False, sigmoid=False):
        """Infer autoencoder architecture (MLP or Convolutional + MLP) from data shape.

        Initialize torch module.

        Args:
            data_shape(tuple[int]): Shape of one sample.
            vae(bool): Make this architecture a VAE.
            sigmoid(bool): Apply sigmoid to decoder output.

        """
        # Infer input size from data. Initialize torch module and optimizer
        if len(data_shape) == 1:
            # Samples are flat vectors. MLP case
            input_size = data_shape[0]
            self.torch_module = AutoencoderModule(input_dim=input_size,
                                                  hidden_dims=self.hidden_dims,
                                                  z_dim=self.n_components,
                                                  noise=self.noise,
                                                  vae=vae,
                                                  sigmoid=sigmoid)
        elif len(data_shape) == 3:
            in_channel, height, width = data_shape
            #  Samples are 3D tensors (i.e. images). Convolutional case.
            self.torch_module = ConvAutoencoderModule(H=height,
                                                      W=width,
                                                      input_channel=in_channel,
                                                      channel_list=self.conv_dims,
                                                      hidden_dims=self.conv_fc_dims,
                                                      z_dim=self.n_components,
                                                      noise=self.noise,
                                                      vae=vae,
                                                      sigmoid=sigmoid)
        else:
            raise Exception(f'Invalid channel number. X has {len(data_shape)}')

        self.torch_module.to(DEVICE)

    def fit(self, x):
        """Fit model to data.

        Args:
            x(BaseDataset): Dataset to fit.

        """

        # Reproducibility
        torch.manual_seed(self.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Save data shape
        self.data_shape = x[0][0].shape

        # Fetch appropriate torch module
        if self.torch_module is None:
            self.init_torch_module(self.data_shape)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.torch_module.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)
        # Train AE
        # Training steps are decomposed as calls to specific methods that can be overriden by children class if need be
        self.torch_module.train()

        self.loader = self.get_loader(x)

        if self.data_val is not None:
            self.val_loader = self.get_loader(self.data_val)

        # Get first metrics
        self.log_metrics(0)

        for epoch in range(1, self.epochs + 1):
            # print(f'            Epoch {epoch}...')
            for batch in self.loader:
                self.optimizer.zero_grad()
                self.train_body(batch)
                self.optimizer.step()

            self.log_metrics(epoch)
            self.end_epoch(epoch)

            # Early stopping
            if self.early_stopping_count == self.patience:
                if self.comet_exp is not None:
                    self.comet_exp.log_metric('early_stopped',
                                              epoch - self.early_stopping_count)
                break

        # Load checkpoint if it exists
        checkpoint_path = os.path.join(self.write_path, 'checkpoint.pt')

        if os.path.exists(checkpoint_path):
            self.load(checkpoint_path)
            os.remove(checkpoint_path)

    def get_loader(self, x):
        """Fetch data loader.

        Args:
            x(BaseDataset): Data to be wrapped in loader.

        Returns:
            torch.utils.data.DataLoader: Torch DataLoader for mini-batch training.

        """
        return torch.utils.data.DataLoader(x, batch_size=self.batch_size, shuffle=True)

    def train_body(self, batch):
        """Called in main training loop to update torch_module parameters.

        Args:
            batch(tuple[torch.Tensor]): Training batch.

        """
        data, _, idx = batch  # No need for labels. Training is unsupervised
        data = data.to(DEVICE)

        x_hat, z = self.torch_module(data)  # Forward pass
        self.compute_loss(data, x_hat, z, idx)

    def compute_loss(self, x, x_hat, z, idx):
        """Apply loss to update parameters following a forward pass.

        Args:
            x(torch.Tensor): Input batch.
            x_hat(torch.Tensor): Reconstructed batch (decoder output).
            z(torch.Tensor): Batch embedding (encoder output).
            idx(torch.Tensor): Indices of samples in batch.

        """
        loss = self.criterion(x_hat, x)
        loss.backward()

    def end_epoch(self, epoch):
        """Method called at the end of every training epoch.

        Args:
            epoch(int): Current epoch.

        """
        pass

    def eval_MSE(self, loader):
        """Compute MSE on data.

        Args:
            loader(DataLoader): Dataset loader.

        Returns:
            float: MSE.

        """
        # Compute MSE over dataset in loader
        self.torch_module.eval()
        sum_loss = 0

        for batch in loader:
            data, _, idx = batch  # No need for labels. Training is unsupervised
            data = data.to(DEVICE)

            x_hat, z = self.torch_module(data)  # Forward pass
            sum_loss += data.shape[0] * self.criterion(data, x_hat).item()

        self.torch_module.train()

        return sum_loss / len(loader.dataset)  # Return average per observation

    def log_metrics(self, epoch):
        """Log metrics.

        Args:
            epoch(int): Current epoch.

        """
        self.log_metrics_train(epoch)
        self.log_metrics_val(epoch)

    def log_metrics_val(self, epoch):
        """Compute validation metrics, log them to comet if need be and update early stopping attributes.

        Args:
            epoch(int):  Current epoch.
        """
        # Validation loss
        if self.val_loader is not None:
            val_mse = self.eval_MSE(self.val_loader)

            if self.comet_exp is not None:
                with self.comet_exp.validate():
                    self.comet_exp.log_metric('MSE_loss', val_mse, epoch=epoch)

            if val_mse < self.current_loss_min:
                # If new min, update attributes and checkpoint model
                self.current_loss_min = val_mse
                self.early_stopping_count = 0
                self.save(os.path.join(self.write_path, 'checkpoint.pt'))
            else:
                self.early_stopping_count += 1

    def log_metrics_train(self, epoch):
        """Log train metrics, log them to comet if need be and update early stopping attributes.

        Args:
            epoch(int):  Current epoch.
        """
        # Train loss
        if self.comet_exp is not None:
            train_mse = self.eval_MSE(self.loader)
            with self.comet_exp.train():
                self.comet_exp.log_metric('MSE_loss', train_mse, epoch=epoch)

    def transform(self, x):
        """Transform data.

        Args:
            x(BaseDataset): Dataset to transform.
        Returns:
            ndarray: Embedding of x.

        """
        self.torch_module.eval()
        loader = torch.utils.data.DataLoader(x, batch_size=self.batch_size,
                                             shuffle=False)
        z = [self.torch_module.encoder(batch.to(DEVICE)).cpu().detach().numpy() for batch, _, _ in loader]
        return np.concatenate(z)

    def inverse_transform(self, x):
        """Take coordinates in the embedding space and invert them to the data space.

        Args:
            x(ndarray): Points in the embedded space with samples on the first axis.
        Returns:
            ndarray: Inverse (reconstruction) of x.

        """
        self.torch_module.eval()
        x = FromNumpyDataset(x)
        loader = torch.utils.data.DataLoader(x, batch_size=self.batch_size,
                                             shuffle=False)
        x_hat = [self.torch_module.decoder(batch.to(DEVICE)).cpu().detach().numpy()
                 for batch in loader]

        return np.concatenate(x_hat)

    def save(self, path):
        """Save state dict.

        Args:
            path(str): File path.

        """
        state = self.torch_module.state_dict()
        state['data_shape'] = self.data_shape
        torch.save(state, path)

    def load(self, path):
        """Load state dict.

        Args:
            path(str): File path.

        """
        state = torch.load(path)
        data_shape = state.pop('data_shape')

        if self.torch_module is None:
            self.init_torch_module(data_shape)

        self.torch_module.load_state_dict(state)


class PHATEAEBase(AE):
    """Standard PHATE AE class.

    AE with geometry regularization. The bottleneck is regularized to match an embedding precomputed by the PHATE manifold
    learning algorithm.
    """

    def __init__(self, *, embedder, embedder_params, lam=100, relax=False, **kwargs):
        """Init.

        Args:
            embedder(BaseModel): Manifold learning class constructor.
            embedder_params(dict): Parameters to pass to embedder.
            lam(float): Regularization factor.
            relax(bool): Use the lambda relaxation scheme. Set to false to use constant lambda throughout training.
            **kwargs: All other arguments with keys are passed to the AE parent class.
        """
        super().__init__(**kwargs)
        self.lam = lam
        self.lam_original = lam  # Needed to compute the lambda relaxation
        self.target_embedding = None  # To store the target embedding as computed by embedder
        self.relax = relax
        self.embedder = embedder(random_state=self.random_state,
                                 n_components=self.n_components,
                                 **embedder_params)  # To compute target embedding.

    def fit(self, x):
        """Fit model to data.

        Args:
            x(BaseDataset): Dataset to fit.

        """
        print('Starting PHATE-AE algorithm ...')
        print('Step 1/2: Approximating the manifold with PHATE ...')
        emb = scipy.stats.zscore(self.embedder.fit_transform(x))  # Normalize embedding
        self.target_embedding = torch.from_numpy(emb).float().to(DEVICE)

        print('Step 2/2: Finetuning the manifold with Autoencoder...')
        super().fit(x)

    def compute_loss(self, x, x_hat, z, idx):
        """Compute torch-compatible geometric loss.

        Args:
            x(torch.Tensor): Input batch.
            x_hat(torch.Tensor): Reconstructed batch (decoder output).
            z(torch.Tensor): Batch embedding (encoder output).
            idx(torch.Tensor): Indices of samples in batch.

        """
        if self.lam > 0:
            loss = self.criterion(x, x_hat) + self.lam * self.criterion(z, self.target_embedding[idx])
        else:
            loss = self.criterion(x, x_hat)

        loss.backward()

    def log_metrics_train(self, epoch):
        """Log train metrics to comet if comet experiment was set.

        Args:
            epoch(int): Current epoch.

        """
        if self.comet_exp is not None:

            # Compute MSE and Geometric Loss over train set
            self.torch_module.eval()
            sum_loss = 0
            sum_geo_loss = 0

            for batch in self.loader:
                data, _, idx = batch  # No need for labels. Training is unsupervised
                data = data.to(DEVICE)

                x_hat, z = self.torch_module(data)  # Forward pass
                sum_loss += data.shape[0] * self.criterion(data, x_hat).item()
                sum_geo_loss += data.shape[0] * self.criterion(z, self.target_embedding[idx]).item()

            with self.comet_exp.train():
                mse_loss = sum_loss / len(self.loader.dataset)
                geo_loss = sum_geo_loss / len(self.loader.dataset)
                self.comet_exp.log_metric('MSE_loss', mse_loss, epoch=epoch)
                self.comet_exp.log_metric('geo_loss', geo_loss, epoch=epoch)
                self.comet_exp.log_metric('GRAE_loss', mse_loss + self.lam * geo_loss, epoch=epoch)
                if self.lam * geo_loss > 0:
                    self.comet_exp.log_metric('geo_on_MSE', self.lam * geo_loss / mse_loss, epoch=epoch)

            self.torch_module.train()

    def end_epoch(self, epoch):
        """Method called at the end of every training epoch.

        Previously used to decay lambda according to the scheme described in the IEEE paper.

        Now using a scheme adapted to early stopping: turn off geometric regularization when reaching 50% of patience

        Args:
            epoch(int): Current epoch.

        """
        if self.relax and self.lam > 0 and self.early_stopping_count == int(self.patience / 2):
            self.lam = 0  # Turn off constraint

            if self.comet_exp is not None:
                self.comet_exp.log_metric('relaxation', epoch, epoch=epoch)


class PHATEAE(PHATEAEBase):
    def __init__(self, *, lam=100, knn=5, gamma=1, t='auto', metric='euclidean', relax=False, **kwargs):
        """Init.

        Args:
            lam(float): Regularization factor.
            knn(int): knn argument of PHATE. Number of neighbors to consider in knn graph.
            t(int): Number of steps of the diffusion operator. Can also be set to 'auto' to select t according to the
            knee point in the Von Neumann Entropy of the diffusion operator
            gamma(float): Informational distance.
            relax(bool): Use the lambda relaxation scheme. Set to false to use constant lambda throughout training.
            **kwargs: All other kehyword arguments are passed to the GRAEBase parent class.
        """
        super().__init__(lam=lam,
                         relax=relax,
                         embedder=PHATE,
                         embedder_params=dict(knn=knn,
                                              t=t,
                                              knn_dist=metric,
                                              knn_max=None, 
                                              mds_dist=metric,
                                              gamma=gamma,
                                              verbose=0,
                                              n_jobs=-1),
                         **kwargs)

