import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

import modules
# from .modules import CausalCNN, Softplus, SqueezeChannels


class FoldVaeClassifFoldWeightZ(nn.Module):
    """
    The VAE method is able to encode a given input into
    mean and log. variance parameters for Gaussian
    distributions that make up a compressed space that represents
    the input. Then, a sample from that space can be decoded into
    an attempted, probably less detailed, reconstruction of the
    original input.
    """
    
    def __init__(
            self, encoder_params, decoder_params, n_split=2, n_class=2, 
            log=print, debug=True):
        super().__init__()
        self.n_split = n_split
        self.log = log
        self.encoder_params = encoder_params
        self.n_class = n_class
        
        self.encoder = CausalCNNVEncoder(**self.encoder_params)
        
        # Adjust decoder split params
        self.decoder_params = decoder_params.copy()
        self.w_split = self.decoder_params['width'] // self.n_split
        self.width_src = self.decoder_params['width']
        self.decoder_params['width'] = self.w_split
        
        self.decoder = CausalCNNVDecoder(**self.decoder_params)
        self.out_sigmoid = nn.Sigmoid()

        self.classif_stage = nn.Sequential(
            nn.Linear(encoder_params['out_channels']*n_split, n_class),
            nn.Softmax(dim=1),
        )
        self.classif_age = nn.Sequential(
            nn.Linear(encoder_params['out_channels']*n_split, 2),
            nn.Softmax(dim=1),
        )

        self.w_folds = nn.Sequential(
            nn.Linear(encoder_params['out_channels'], 1),
            nn.Softplus(),
        )
        # self.classif_folds = nn.Sequential(
        #     nn.Linear(encoder_params['out_channels'], n_class),
        #     nn.Softmax(dim=2),
        # )

        self.debug = debug
        
        self.kl = 0
    
    def reparameterize(self, mu, sd):
        """Sample from a Gaussian distribution with given mean and s.d."""
        # eps = torch.randn_like(sd)
        eps = torch.normal(torch.zeros_like(mu), torch.ones_like(sd))
        return mu + eps * sd
    
    def forward(self, x):
        """Return reconstructed input after compressing it"""
        self.log(f"input: {x.shape}") if self.debug else None
        z = None
        x_hat = None
        enc_mu, enc_sd = None, None
        z_folds = None
        for i_split in range(self.n_split):
            x_split = x[:, :, i_split*self.w_split:(i_split+1)*self.w_split]

            # encoding
            _enc_mu, _enc_sd = self.encoder(x_split)
            self.log(f"--VAE enc_mu:{_enc_mu.shape}, enc_sd:{_enc_sd.shape}") if self.debug else None
            _z = self.reparameterize(_enc_mu, _enc_sd)
            r"Add all _enc_mu and _enc_sd to do average later."
            _enc_mu = _enc_mu.view(_enc_mu.size(0), 1, -1)
            _enc_sd = _enc_sd.view(_enc_sd.size(0), 1, -1)
            if enc_mu is None:
                enc_mu = _enc_mu
                enc_sd = _enc_sd
            else:
                enc_mu = torch.cat((enc_mu, _enc_mu), dim=1)
                enc_sd = torch.cat((enc_sd, _enc_sd), dim=1)
            self.log(f"--VAE _z:{_z.shape}, enc_mu:{enc_mu.shape}, enc_sd:{enc_sd.shape}") if self.debug else None
            if z is None:
                z = _z
                r"(B, Fold#, Z_fold)"
                # z_folds = torch.unsqueeze(_z, 1)
            else:
                z = torch.cat((z, _z), dim=1)
                # z_folds = torch.cat((z_folds, torch.unsqueeze(_z, 1)), dim=1)
            self.log(f"ENC x_split[{i_split}]:{x_split.size()}, _z:{_z.size()}, z:{z.size()}") if self.debug else None

            # decoding
            _x_hat = self.decoder(_z)
            _x_hat = self.out_sigmoid(_x_hat)
            if x_hat is None:
                x_hat = _x_hat
            else:
                x_hat = torch.cat((x_hat, _x_hat), dim=2)
            self.log(f"DEC x_split[{i_split}]:{x_split.size()}, _x_hat:{_x_hat.size()}, x_hat:{x_hat.size()}") if self.debug else None

        # z_com = self.z_compress(z)

        r"Average enc_mu and enc_sd"
        enc_mu = torch.mean(enc_mu, 1)
        enc_sd = torch.mean(enc_sd, 1)
        self.log(f"Aggregate enc_mu:{enc_mu.shape}, enc_sd:{enc_sd.shape}") if self.debug else None
        # calculate kl loss
        # 
        sigma = enc_sd
        # self.log(f"z: {z.shape}") if self.debug else None
      
        self.kl = -0.5 * torch.sum(1 + sigma - enc_mu.pow(2) - sigma.exp(), dim=1)
        self.kl = self.kl.mean()

        "Voted classif output"
        # w_folds = self.w_folds(z_folds)
        # self.log(f"w_folds: {w_folds.shape}") if self.debug else None
        
        # z_folds = z_folds * w_folds
        # self.log(f"z_folds: {z_folds.shape}") if self.debug else None
        
        # z_folds = z_folds.view(z_folds.size(0), -1)
        # self.log(f"z_folds_flat: {z_folds.shape}") if self.debug else None
        
        # cls_proba_folds = self.classif_folds(z_folds)
        # proba_idx_folds = torch.argmax(cls_proba_folds, dim=2).to(x.device)
        # bat_bc = self.batched_bincount(proba_idx_folds, 1, self.n_class).to(x.device)
        # cls_voted = torch.argmax(bat_bc, dim=1).to(x.device)
        # cls_proba_voted = bat_bc.type('torch.FloatTensor')
        # cls_proba_voted = F.softmax(bat_bc.type('torch.FloatTensor'), dim=1)

        if self.debug:
            self.debug = not self.debug

        return {
            'x_hat':x_hat, 
            'z':z, 
            'clz_proba': self.classif_stage(z),
            'clz_proba_age': self.classif_age(z),            
            # 'clz_proba': self.classif_stage(z_folds),
            # 'clz_proba_age': self.classif_age(z_folds),
            # 'w_folds': w_folds.detach().cpu().numpy()
        }
        # return x_hat, z, self.classif_stage(z), self.classif_age(z)

    def batched_bincount(self, x, dim, max_value):
        target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
        values = torch.ones_like(x)
        target.scatter_add_(dim, x, values)
        return target


class FoldVaeClassifFoldWeight(nn.Module):
    """
    The VAE method is able to encode a given input into
    mean and log. variance parameters for Gaussian
    distributions that make up a compressed space that represents
    the input. Then, a sample from that space can be decoded into
    an attempted, probably less detailed, reconstruction of the
    original input.
    """
    
    def __init__(
            self, encoder_params, decoder_params, n_split=2, n_class=2, 
            log=print, debug=True):
        super().__init__()
        self.n_split = n_split
        self.log = log
        self.encoder_params = encoder_params
        self.n_class = n_class
        
        self.encoder = CausalCNNVEncoder(**self.encoder_params)
        
        # Adjust decoder split params
        self.decoder_params = decoder_params.copy()
        self.w_split = self.decoder_params['width'] // self.n_split
        self.width_src = self.decoder_params['width']
        self.decoder_params['width'] = self.w_split
        
        self.decoder = CausalCNNVDecoder(**self.decoder_params)
        self.out_sigmoid = nn.Sigmoid()

        self.classif_stage = nn.Sequential(
            nn.Linear(encoder_params['out_channels']*n_split, n_class),
            nn.Softmax(dim=1),
        )
        self.classif_age = nn.Sequential(
            nn.Linear(encoder_params['out_channels']*n_split, 2),
            nn.Softmax(dim=1),
        )

        self.w_folds = nn.Sequential(
            nn.Linear(encoder_params['out_channels'], 1),
            # nn.Softplus(),
            nn.Sigmoid()
        )
        # self.classif_folds = nn.Sequential(
        #     nn.Linear(encoder_params['out_channels'], n_class),
        #     nn.Softmax(dim=2),
        # )

        self.debug = debug
        
        self.kl = 0
    
    def reparameterize(self, mu, sd):
        """Sample from a Gaussian distribution with given mean and s.d."""
        # eps = torch.randn_like(sd)
        eps = torch.normal(torch.zeros_like(mu), torch.ones_like(sd))
        return mu + eps * sd
    
    def forward(self, x):
        """Return reconstructed input after compressing it"""
        self.log(f"input: {x.shape}") if self.debug else None
        z = None
        x_hat = None
        enc_mu, enc_sd = None, None
        z_folds = None
        for i_split in range(self.n_split):
            x_split = x[:, :, i_split*self.w_split:(i_split+1)*self.w_split]

            # encoding
            _enc_mu, _enc_sd = self.encoder(x_split)
            self.log(f"--VAE enc_mu:{_enc_mu.shape}, enc_sd:{_enc_sd.shape}") if self.debug else None
            _z = self.reparameterize(_enc_mu, _enc_sd)
            r"Add all _enc_mu and _enc_sd to do average later."
            _enc_mu = _enc_mu.view(_enc_mu.size(0), 1, -1)
            _enc_sd = _enc_sd.view(_enc_sd.size(0), 1, -1)
            if enc_mu is None:
                enc_mu = _enc_mu
                enc_sd = _enc_sd
            else:
                enc_mu = torch.cat((enc_mu, _enc_mu), dim=1)
                enc_sd = torch.cat((enc_sd, _enc_sd), dim=1)
            self.log(f"--VAE _z:{_z.shape}, enc_mu:{enc_mu.shape}, enc_sd:{enc_sd.shape}") if self.debug else None
            if z is None:
                z = _z
                r"(B, Fold#, Z_fold)"
                z_folds = torch.unsqueeze(_z, 1)
            else:
                z = torch.cat((z, _z), dim=1)
                z_folds = torch.cat((z_folds, torch.unsqueeze(_z, 1)), dim=1)
            self.log(f"ENC x_split[{i_split}]:{x_split.size()}, _z:{_z.size()}, z:{z.size()}") if self.debug else None

            # decoding
            _x_hat = self.decoder(_z)
            _x_hat = self.out_sigmoid(_x_hat)
            if x_hat is None:
                x_hat = _x_hat
            else:
                _hat = torch.cat((x_hat, _x_hat), dim=2)
            self.log(f"DEC x_split[{i_split}]:{x_split.size()}, _x_hat:{_x_hat.size()}, x_hat:{x_hat.size()}") if self.debug else None

        # z_com = self.z_compress(z)

        r"Average enc_mu and enc_sd"
        enc_mu = torch.mean(enc_mu, 1)
        enc_sd = torch.mean(enc_sd, 1)
        self.log(f"Aggregate enc_mu:{enc_mu.shape}, enc_sd:{enc_sd.shape}") if self.debug else None
        # calculate kl loss
        # 
        sigma = enc_sd
        # self.log(f"z: {z.shape}") if self.debug else None
      
        self.kl = -0.5 * torch.sum(1 + sigma - enc_mu.pow(2) - sigma.exp(), dim=1)
        self.kl = self.kl.mean()

        "Voted classif output"
        w_folds = self.w_folds(z_folds)
        self.log(f"w_folds: {w_folds.shape}") if self.debug else None
        
        z_folds = z_folds * w_folds
        self.log(f"z_folds: {z_folds.shape}") if self.debug else None
        
        z_folds = z_folds.view(z_folds.size(0), -1)
        self.log(f"z_folds_flat: {z_folds.shape}") if self.debug else None
        
        # cls_proba_folds = sel.classif_folds(z_folds)
        # proba_idx_folds = torch.argmax(cls_proba_folds, dim=2).to(x.device)
        # bat_bc = self.batched_bincount(proba_idx_folds, 1, self.n_class).to(x.device)
        # cls_voted = torch.argmax(bat_bc, dim=1).to(x.device)
        # cls_proba_voted = bat_bc.type('torch.FloatTensor')
        # cls_proba_voted = F.softmax(bat_bc.type('torch.FloatTensor'), dim=1)

        if self.debug:
            self.debug = not self.debug

        return {
            'x_hat':x_hat, 
            'z':z, 
            # 'clz_idx_folds': proba_idx_folds,
            # 'clz_voted': cls_voted,
            # 'clz_proba_voted': cls_proba_voted,
            'clz_proba': self.classif_stage(z_folds),
            'clz_proba_age': self.classif_age(z_folds),
            'w_folds': w_folds.detach().cpu().numpy()}
        # return x_hat, z, self.classif_stage(z), self.classif_age(z)

    def batched_bincount(self, x, dim, max_value):
        target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
        values = torch.ones_like(x)
        target.scatter_add_(dim, x, values)
        return target

class FoldVaeClassifFoldWeightParameterizer(nn.Module):
    """
    The VAE method is able to encode a given input into
    mean and log. variance parameters for Gaussian
    distributions that make up a compressed space that represents
    the input. Then, a sample from that space can be decoded into
    an attempted, probably less detailed, reconstruction of the
    original input.
    """

    def __init__(
            self, encoder_params, decoder_params, n_split=2, n_class=2, 
            log=print, debug=True):
        super().__init__()
        self.n_split = n_split
        self.log = log
        self.encoder_params = encoder_params
        self.n_class = n_class
        
        self.encoder = CausalCNNVEncoder(**self.encoder_params)
        
        # Adjust decoder split params
        self.decoder_params = decoder_params.copy()
        self.w_split = self.decoder_params['width'] // self.n_split
        self.width_src = self.decoder_params['width']
        self.decoder_params['width'] = self.w_split
        
        self.decoder = CausalCNNVDecoder(**self.decoder_params)
        self.out_sigmoid = nn.Sigmoid()

        self.parameterizer = Parameterizer(self.encoder_params["in_channels"], 32, 5, n_split)
        self.aggregator = Aggregator(32, n_class)
        self.classif_stage = nn.Softmax(dim=1)

        self.debug = debug
        
        self.kl = 0
    
    def reparameterize(self, mu, sd):
        """Sample from a Gaussian distribution with given mean and s.d."""
        eps = torch.normal(torch.zeros_like(mu), torch.ones_like(sd))
        return mu + eps * sd
    
    def forward(self, x):
        """Return reconstructed input after compressing it"""
        self.log(f"input: {x.shape}") if self.debug else None
        z = None
        x_hat = None
        enc_mu, enc_sd = None, None
        z_folds = None
        for i_split in range(self.n_split):
            x_split = x[:, :, i_split*self.w_split:(i_split+1)*self.w_split]

            # encoding
            _enc_mu, _enc_sd = self.encoder(x_split)
            self.log(f"--VAE enc_mu:{_enc_mu.shape}, enc_sd:{_enc_sd.shape}") if self.debug else None
            _z = self.reparameterize(_enc_mu, _enc_sd)
            r"Add all _enc_mu and _enc_sd to do average later."
            _enc_mu = _enc_mu.view(_enc_mu.size(0), 1, -1)
            _enc_sd = _enc_sd.view(_enc_sd.size(0), 1, -1)
            if enc_mu is None:
                enc_mu = _enc_mu
                enc_sd = _enc_sd
            else:
                enc_mu = torch.cat((enc_mu, _enc_mu), dim=1)
                enc_sd = torch.cat((enc_sd, _enc_sd), dim=1)
            self.log(f"--VAE _z:{_z.shape}, enc_mu:{enc_mu.shape}, enc_sd:{enc_sd.shape}") if self.debug else None
            if z is None:
                z = _z
                r"(B, Fold#, Z_fold)"
                z_folds = torch.unsqueeze(_z, 1)
            else:
                z = torch.cat((z, _z), dim=1)
                z_folds = torch.cat((z_folds, torch.unsqueeze(_z, 1)), dim=1)
            self.log(f"ENC x_split[{i_split}]:{x_split.size()}, _z:{_z.size()}, z:{z.size()}") if self.debug else None

            # decoding
            _x_hat = self.decoder(_z)
            _x_hat = self.out_sigmoid(_x_hat)
            if x_hat is None:
                x_hat = _x_hat
            else:
                x_hat = torch.cat((x_hat, _x_hat), dim=2)
            self.log(f"DEC x_split[{i_split}]:{x_split.size()}, _x_hat:{_x_hat.size()}, x_hat:{x_hat.size()}") if self.debug else None

        r"Average enc_mu and enc_sd"
        enc_mu = torch.mean(enc_mu, 1)
        enc_sd = torch.mean(enc_sd, 1)
        self.log(f"Aggregate enc_mu:{enc_mu.shape}, enc_sd:{enc_sd.shape}") if self.debug else None
        # calculate kl loss
        # 
        sigma = enc_sd
        # self.log(f"z: {z.shape}") if self.debug else None
      
        self.kl = -0.5 * torch.sum(1 + sigma - enc_mu.pow(2) - sigma.exp(), dim=1)
        self.kl = self.kl.mean()

        # Calculate relevance weights on signal
        relevance_weights = self.parameterizer(x)
        
        # Aggregate folds and relevance weights
        out = self.aggregator(z_folds, relevance_weights)
        
        z_folds = z_folds.view(z_folds.size(0), -1)
        self.log(f"z_folds_flat: {z_folds.shape}") if self.debug else None

        if self.debug:
            self.debug = not self.debug

        return {
            'x_hat':x_hat, 
            'z':z,
            'clz_proba': self.classif_stage(out),
            'w_folds': relevance_weights.detach().cpu().numpy()
        }

    def batched_bincount(self, x, dim, max_value):
        target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
        values = torch.ones_like(x)
        target.scatter_add_(dim, x, values)
        return target


class FoldVaeClassif(nn.Module):
    """
    The VAE method is able to encode a given input into
    mean and log. variance parameters for Gaussian
    distributions that make up a compressed space that represents
    the input. Then, a sample from that space can be decoded into
    an attempted, probably less detailed, reconstruction of the
    original input.
    """
    
    def __init__(
            self, encoder_params, decoder_params, n_split=2, n_class=2, 
            log=print, debug=True):
        super().__init__()
        self.n_split = n_split
        self.log = log
        self.encoder_params = encoder_params
        
        self.encoder = CausalCNNVEncoder(**self.encoder_params)
        
        # Adjust decoder split params
        self.decoder_params = decoder_params.copy()
        self.w_split = self.decoder_params['width'] // self.n_split
        self.width_src = self.decoder_params['width']
        self.decoder_params['width'] = self.w_split
        
        self.decoder = CausalCNNVDecoder(**self.decoder_params)
        self.out_sigmoid = nn.Sigmoid()

        self.classif_stage = nn.Sequential(
            nn.Linear(encoder_params['out_channels']*n_split, n_class),
            nn.Softmax(dim=1),
        )
        self.classif_age = nn.Sequential(
            nn.Linear(encoder_params['out_channels']*n_split, 2),
            nn.Softmax(dim=1),
        )
        # self.z_compress = nn.Sequential(
        #     nn.Linear(encoder_params['out_channels']*n_split, encoder_params['out_channels']),
        #     nn.BatchNorm1d(num_features=encoder_params['out_channels']), nn.ReLU(), 
        #     # nn.Dropout(0.2)
        #     # nn.Softmax(dim=1),  
        #     # nn.ReLU(),
        # ) 

        self.debug = debug
        
        self.kl = 0
    
    def reparameterize(self, mu, sd):
        """Sample from a Gaussian distribution with given mean and s.d."""
        # eps = torch.randn_like(sd)
        eps = torch.normal(torch.zeros_like(mu), torch.ones_like(sd))
        return mu + eps * sd
    
    def forward(self, x):
        """Return reconstructed input after compressing it"""
        self.log(f"input: {x.shape}") if self.debug else None
        z = None
        x_hat = None
        enc_mu, enc_sd = None, None
        for i_split in range(self.n_split):
            x_split = x[:, :, i_split*self.w_split:(i_split+1)*self.w_split]

            # encoding
            _enc_mu, _enc_sd = self.encoder(x_split)
            self.log(f"--VAE enc_mu:{_enc_mu.shape}, enc_sd:{_enc_sd.shape}") if self.debug else None
            _z = self.reparameterize(_enc_mu, _enc_sd)
            r"Add all _enc_mu and _enc_sd to do average later."
            _enc_mu = _enc_mu.view(_enc_mu.size(0), 1, -1)
            _enc_sd = _enc_sd.view(_enc_sd.size(0), 1, -1)
            if enc_mu is None:
                enc_mu = _enc_mu
                enc_sd = _enc_sd
            else:
                enc_mu = torch.cat((enc_mu, _enc_mu), dim=1)
                enc_sd = torch.cat((enc_sd, _enc_sd), dim=1)
            self.log(f"--VAE _z:{_z.shape}, enc_mu:{enc_mu.shape}, enc_sd:{enc_sd.shape}") if self.debug else None
            if z is None:
                z = _z
            else:
                z = torch.cat((z, _z), dim=1)
            self.log(f"ENC x_split[{i_split}]:{x_split.size()}, _z:{_z.size()}, z:{z.size()}") if self.debug else None

            # decoding
            _x_hat = self.decoder(_z)
            _x_hat = self.out_sigmoid(_x_hat)
            if x_hat is None:
                x_hat = _x_hat
            else:
                x_hat = torch.cat((x_hat, _x_hat), dim=2)
            self.log(f"DEC x_split[{i_split}]:{x_split.size()}, _x_hat:{_x_hat.size()}, x_hat:{x_hat.size()}") if self.debug else None

        # z_com = self.z_compress(z)

        r"Average enc_mu and enc_sd"
        enc_mu = torch.mean(enc_mu, 1)
        enc_sd = torch.mean(enc_sd, 1)
        self.log(f"Aggregate enc_mu:{enc_mu.shape}, enc_sd:{enc_sd.shape}") if self.debug else None
        # calculate kl loss
        # 
        sigma = enc_sd
        self.log(f"   z: {z.shape}") if self.debug else None
      
        self.kl = -0.5 * torch.sum(1 + sigma - enc_mu.pow(2) - sigma.exp(), dim=1)
        self.kl = self.kl.mean()

        if self.debug:
            self.debug = not self.debug

        # return x_hat, z, cls_proba, (enc_mu, enc_sd)
        # return x_hat, z, z_com, (enc_mu, enc_sd)
        return x_hat, z, self.classif_stage(z), self.classif_age(z)
    

class FoldVAE(nn.Module):
    """
    The VAE method is able to encode a given input into
    mean and log. variance parameters for Gaussian
    distributions that make up a compressed space that represents
    the input. Then, a sample from that space can be decoded into
    an attempted, probably less detailed, reconstruction of the
    original input.
    """
    
    def __init__(
            self, encoder_params, decoder_params, n_split=2, n_class=2, 
            log=print, debug=True):
        super().__init__()
        self.n_split = n_split
        self.log = log
        self.encoder_params = encoder_params
        
        self.encoder = CausalCNNVEncoder(**self.encoder_params)
        
        # Adjust decoder split params
        self.decoder_params = decoder_params.copy()
        self.w_split = self.decoder_params['width'] // self.n_split
        self.width_src = self.decoder_params['width']
        self.decoder_params['width'] = self.w_split
        
        self.decoder = CausalCNNVDecoder(**self.decoder_params)
        self.out_sigmoid = nn.Sigmoid()

        self.z_compress = nn.Sequential(
            nn.Linear(encoder_params['out_channels']*n_split, encoder_params['out_channels']),
            nn.BatchNorm1d(num_features=encoder_params['out_channels']), nn.ReLU(), 
            # nn.Dropout(0.2)
            # nn.Softmax(dim=1),  
            # nn.ReLU(),
        ) 

        self.debug = debug
        
        self.kl = 0
    
    def reparameterize(self, mu, sd):
        """Sample from a Gaussian distribution with given mean and s.d."""
        # eps = torch.randn_like(sd)
        eps = torch.normal(torch.zeros_like(mu), torch.ones_like(sd))
        return mu + eps * sd
    
    def forward(self, x):
        """Return reconstructed input after compressing it"""
        self.log(f"input: {x.shape}") if self.debug else None
        z = None
        x_hat = None
        enc_mu, enc_sd = None, None
        for i_split in range(self.n_split):
            x_split = x[:, :, i_split*self.w_split:(i_split+1)*self.w_split]

            # encoding
            _enc_mu, _enc_sd = self.encoder(x_split)
            self.log(f"--VAE enc_mu:{_enc_mu.shape}, enc_sd:{_enc_sd.shape}") if self.debug else None
            _z = self.reparameterize(_enc_mu, _enc_sd)
            r"Add all _enc_mu and _enc_sd to do average later."
            _enc_mu = _enc_mu.view(_enc_mu.size(0), 1, -1)
            _enc_sd = _enc_sd.view(_enc_sd.size(0), 1, -1)
            if enc_mu is None:
                enc_mu = _enc_mu
                enc_sd = _enc_sd
            else:
                enc_mu = torch.cat((enc_mu, _enc_mu), dim=1)
                enc_sd = torch.cat((enc_sd, _enc_sd), dim=1)
            self.log(f"--VAE _z:{_z.shape}, enc_mu:{enc_mu.shape}, enc_sd:{enc_sd.shape}") if self.debug else None
            if z is None:
                z = _z
            else:
                z = torch.cat((z, _z), dim=1)
            self.log(f"ENC x_split[{i_split}]:{x_split.size()}, _z:{_z.size()}, z:{z.size()}") if self.debug else None

            # decoding
            _x_hat = self.decoder(_z)
            _x_hat = self.out_sigmoid(_x_hat)
            if x_hat is None:
                x_hat = _x_hat
            else:
                x_hat = torch.cat((x_hat, _x_hat), dim=2)
            self.log(f"DEC x_split[{i_split}]:{x_split.size()}, _x_hat:{_x_hat.size()}, x_hat:{x_hat.size()}") if self.debug else None

        z_com = self.z_compress(z)

        r"Average enc_mu and enc_sd"
        enc_mu = torch.mean(enc_mu, 1)
        enc_sd = torch.mean(enc_sd, 1)
        self.log(f"Aggregate enc_mu:{enc_mu.shape}, enc_sd:{enc_sd.shape}") if self.debug else None
        # calculate kl loss
        # 
        sigma = enc_sd
        self.log(f"   z: {z.shape}") if self.debug else None
      
        self.kl = -0.5 * torch.sum(1 + sigma - enc_mu.pow(2) - sigma.exp(), dim=1)
        self.kl = self.kl.mean()

        if self.debug:
            self.debug = not self.debug

        # return x_hat, z, cls_proba, (enc_mu, enc_sd)
        # return x_hat, z, z_com, (enc_mu, enc_sd)
        return x_hat, z, z_com, (enc_mu, enc_sd)
    

class VAE(nn.Module):
    """
    The VAE method is able to encode a given input into
    mean and log. variance parameters for Gaussian
    distributions that make up a compressed space that represents
    the input. Then, a sample from that space can be decoded into
    an attempted, probably less detailed, reconstruction of the
    original input.
    """
    
    def __init__(self, encoder_params, decoder_params, debug=False):
        super().__init__()
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.encoder = CausalCNNVEncoder(**encoder_params)
        self.decoder = CausalCNNVDecoder(**decoder_params)
        self.out_sigmoid = nn.Sigmoid()
        self.debug = debug
        
        self.kl = 0
    
    def reparameterize(self, mu, sd):
        """Sample from a Gaussian distribution with given mean and s.d."""
        # eps = torch.randn_like(sd)
        eps = torch.normal(torch.zeros_like(mu), torch.ones_like(sd))
        return mu + eps * sd
    
    def forward(self, x):
        """Return reconstructed input after compressing it"""
        enc_mu, enc_sd = self.encoder(x)
        print(f"--VAE enc_mu:{enc_mu.shape}, enc_sd:{enc_sd.shape}") if self.debug else None
        z = self.reparameterize(enc_mu, enc_sd)
        print(f"--VAE z:{z.shape}") if self.debug else None
        if self.decoder_params['gaussian_out']:
            dec_mu, dec_sd = self.decoder(z)
            # reconstruction is equal to the mean value for the gauss. distr. of each point
            # reshape the mean vector to be of same size as input (Bx8x600 for median ECGs)
            recon_x = dec_mu.view(x.shape)
            print(f"--VAE recon_x:{recon_x.shape}") if self.debug else None
            return recon_x, z, [(enc_mu, enc_sd), (dec_mu, dec_sd)]
        recon_x = self.decoder(z)
        recon_x = self.out_sigmoid(recon_x)
        print(f"--VAE d(z) recon_x:{recon_x.shape}") if self.debug else None


        # calculate kl loss
        # 
        sigma = enc_sd
        # print(f"   sigma: {sigma.shape}") if self.debug else None
        # sigma_exp = torch.exp(sigma)
        # print(f"   sigma-exp: {sigma_exp.shape}") if self.debug else None

        # epsilon = torch.distributions.Normal(0, 1).sample(sigma.shape).to(sigma.device)
        # z = enc_mu + sigma_exp * epsilon
        # z = mu + sigma*self.N.sample(mu.shape)
        print(f"   z: {z.shape}") if self.debug else None

        # self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()        
        self.kl = -0.5 * torch.sum(1 + sigma - enc_mu.pow(2) - sigma.exp(), dim=1)
        self.kl = self.kl.mean()

        if self.debug:
            self.debug = not self.debug

        return recon_x, z, (enc_mu, enc_sd)    


class CausalCNNEncoder(torch.nn.Module):
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).

    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of channels manipulated in the causal CNN.
        depth (int): Depth of the causal CNN.
        reduced_size (int): Fixed length to which the output time series of the
           causal CNN is reduced.
        out_channels (int): Number of output classes.
        kernel_size (int): Kernel size of the applied non-residual convolutions.
        dropout (float): The dropout probability between 0 and 1.
    """
    def __init__(self, in_channels, channels, depth, reduced_size,
                 out_channels, kernel_size, dropout):
        super(CausalCNNEncoder, self).__init__()
        causal_cnn = modules.CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        spatial = modules.Spatial(reduced_size, dropout)
        reduce_size = torch.nn.AdaptiveAvgPool1d(1)
        squeeze = modules.SqueezeChannels()  # Squeezes the third dimension (time)
        linear1 = torch.nn.Linear(reduced_size, 26)
        linear2 = torch.nn.Linear(26, out_channels)
        self.network = torch.nn.Sequential(
            causal_cnn, spatial, reduce_size, squeeze, linear1,
            nn.BatchNorm1d(num_features=26), nn.ReLU(), nn.Dropout(dropout), linear2,
        )

    def forward(self, x):
        return self.network(x)


class CausalCNNVEncoder(torch.nn.Module):
    """
    Variational encoder. Difference is that we need two outputs: mean and
    standard deviation.

    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of channels manipulated in the causal CNN.
        depth (int): Depth of the causal CNN.
        reduced_size (int): Fixed length to which the output time series of the
           causal CNN is reduced.
        out_channels (int): Number of output classes.
        kernel_size (int): Kernel size of the applied non-residual convolutions.
        softplus_eps (float): Small number to add for stability of the Softplus activation.
        dropout (float): The dropout probability between 0 and 1.
        sd_output (bool): Put to true when using this class inside a VAE, as
            an additional output for the SD is added.
    """
    def __init__(self, in_channels: int, channels: int, depth: int, reduced_size: int,
                 out_channels: int, kernel_size: int, softplus_eps: float, dropout: float, 
                 sd_output: bool = True):
        super(CausalCNNVEncoder, self).__init__()
        causal_cnn = modules.CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = modules.SqueezeChannels()  # Squeezes the third dimension (time)
        self.network = torch.nn.Sequential(
            causal_cnn, reduce_size, squeeze,
        )
        self.linear_mean = torch.nn.Linear(reduced_size, out_channels)
        self.sd_output = sd_output
        if self.sd_output:
            self.linear_sd = torch.nn.Sequential(
                torch.nn.Linear(reduced_size, out_channels),
                modules.Softplus(softplus_eps),
            )

    def forward(self, x):
        out = self.network(x)
        if self.sd_output:
            return self.linear_mean(out), self.linear_sd(out)
        return self.linear_mean(out).squeeze()
    

class CausalCNNVDecoder(torch.nn.Module):
    """
    Variational decoder.
    """
    def __init__(self, k, width, in_channels, channels, depth, out_channels,
                 kernel_size, gaussian_out, softplus_eps, dropout, debug=True):
        super(CausalCNNVDecoder, self).__init__()
        self.in_channels = in_channels
        self.width = width
        self.gaussian_out = gaussian_out
        self.linear1 = torch.nn.Linear(k, in_channels)
        self.linear2 = torch.nn.Linear(in_channels, in_channels * width)
        self.causal_cnn = modules.CausalCNN(
            in_channels, channels, depth, out_channels, kernel_size,
            forward=False,
        )
        if self.gaussian_out:
            self.linear_mean = nn.Linear(out_channels * width, 
                                         out_channels * width)
            self.linear_sd = torch.nn.Sequential(
                torch.nn.Linear(out_channels * width, out_channels * width),
                modules.Softplus(softplus_eps),
            )
        self.debug = debug
        
    def forward(self, x):
        """
        Returns a reconstruction of the original 8x600 ECG, by decoding
        the given compression.
        """
        B, _ = x.shape
        # from (BxK) to (BxC)
        out = self.linear1(x)
        print(f"--CausalCNNVDecoder linear1:{out.shape}") if self.debug else None
        # from (BxC) to (Bx(C*600))
        out = self.linear2(out)
        print(f"--CausalCNNVDecoder linear2:{out.shape}") if self.debug else None
        # from (Bx(C*600)) to (BxCx600)
        out = out.view(B, self.in_channels, self.width)
        print(f"--CausalCNNVDecoder reshape:{out.shape}") if self.debug else None
        # deconvolve through the causal CNN
        out = self.causal_cnn(out)
        print(f"--CausalCNNVDecoder out_causal_cnn:{out.shape}") if self.debug else None

        if self.debug:
            self.debug = not self.debug

        if self.gaussian_out:
            nflat_shape = out.shape
            # flatten the output to shape (Bx(8*600))
            out = torch.flatten(out, start_dim=1)
            return self.linear_mean(out).reshape(nflat_shape), self.linear_sd(out).reshape(nflat_shape)
        return out
    

class FoldCausalCNNVEncoder(torch.nn.Module):
    """
    Variational encoder. Difference is that we need two outputs: mean and
    standard deviation.

    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of channels manipulated in the causal CNN.
        depth (int): Depth of the causal CNN.
        reduced_size (int): Fixed length to which the output time series of the
           causal CNN is reduced.
        out_channels (int): Number of output classes.
        kernel_size (int): Kernel size of the applied non-residual convolutions.
        softplus_eps (float): Small number to add for stability of the Softplus activation.
        dropout (float): The dropout probability between 0 and 1.
        sd_output (bool): Put to true when using this class inside a VAE, as
            an additional output for the SD is added.
    """
    def __init__(self, in_channels: int, channels: int, depth: int, reduced_size: int,
                 out_channels: int, kernel_size: int, softplus_eps: float, dropout: float, 
                 sd_output: bool = True, input_split: int = 4):
        super(FoldCausalCNNVEncoder, self).__init__()
        causal_cnn = modules.CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = modules.SqueezeChannels()  # Squeezes the third dimension (time)
        self.network = torch.nn.Sequential(
            causal_cnn, reduce_size, squeeze,
        )
        self.linear_mean = torch.nn.Linear(reduced_size, out_channels)
        self.sd_output = sd_output
        if self.sd_output:
            self.linear_sd = torch.nn.Sequential(
                torch.nn.Linear(reduced_size, out_channels),
                modules.Softplus(softplus_eps),
            )

    def forward(self, x):
        out = self.network(x)
        if self.sd_output:
            return self.linear_mean(out), self.linear_sd(out)
        return self.linear_mean(out).squeeze()
    

class ConvNormPool(nn.Module):
    """Conv Skip-connection module"""

    def __init__(self, input_size, hidden_size, kernel_size, norm_type="batchnorm"):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv1d(
            in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size
        )
        self.conv_2 = nn.Conv1d(
            in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size
        )
        self.conv_3 = nn.Conv1d(
            in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size
        )
        self.swish_1 = modules.Swish()
        self.swish_2 = modules.Swish()
        self.swish_3 = modules.Swish()
        if norm_type == "group":
            self.normalization_1 = nn.GroupNorm(num_groups=8, num_channels=hidden_size)
            self.normalization_2 = nn.GroupNorm(num_groups=8, num_channels=hidden_size)
            self.normalization_3 = nn.GroupNorm(num_groups=8, num_channels=hidden_size)
        else:
            self.normalization_1 = nn.BatchNorm1d(num_features=hidden_size)
            self.normalization_2 = nn.BatchNorm1d(num_features=hidden_size)
            self.normalization_3 = nn.BatchNorm1d(num_features=hidden_size)

        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, input):
        conv1 = self.conv_1(input)
        x = self.normalization_1(conv1)
        x = self.swish_1(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))

        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = self.swish_2(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))

        conv3 = self.conv_3(x)
        x = self.normalization_3(conv1 + conv3)
        x = self.swish_3(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))

        x = self.pool(x)
        return x


class Parameterizer(nn.Module):

    def __init__(self, d_input, d_hidden, d_kernel, n_split, dropout=0.5):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvNormPool(d_input, d_hidden, d_kernel),
            nn.AdaptiveMaxPool1d((1))
        )

        self.proj = nn.Sequential(
            nn.BatchNorm1d(d_hidden),
            modules.Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_split),
            nn.Softplus(),
        )

    def forward(self, x):
        
        Batch = x.shape[0]
        
        encoded_sequence = self.encoder(x).reshape(Batch, -1)
        relevance_weights = self.proj(encoded_sequence)

        return relevance_weights
    
class Aggregator(nn.Module):
    def __init__(self, d_hidden, d_out):
        super().__init__()

        self.fc = nn.Sequential(
            nn.BatchNorm1d(d_hidden),
            modules.Swish(),
            nn.Linear(d_hidden, d_out)
        )

    def forward(self, z_folds, relevance_weights, ignore_relevance_weights=False):

        (Batch, Folds) = relevance_weights.shape

        out = self.fc(z_folds.reshape(Batch * Folds, -1))
        if not ignore_relevance_weights:
            out = (out.reshape(Batch, Folds, -1) * relevance_weights.unsqueeze(-1)).sum(
                    1
                )
        return out


def main():
    CFG_FILE = 'config_multimodal_ecg.yml'
    with open(CFG_FILE, 'r') as stream:
        params = yaml.safe_load(stream)
        params['seg_len'] = params['hz'] * params['seg_len_sec']
        params['decoder']['width'] = params['seg_len']

    class_map = {0:0, 1:1, 2:1, 3:1, 4:1, 5:2}
    n_class = len(set(class_map.values()))
    params['n_class'] = n_class

    print(params)

    params['n_class'] = n_class
    params_decoder = params['decoder'].copy()
    params_decoder['width'] = params['hz'] * params['seg_len_sec']
    net = FoldVaeClassifFoldWeight(
        params['encoder'], params_decoder, n_split=params['n_split'], 
        n_class=params['n_class'], debug=True,
    )

    x = torch.randn(3, 1, params['seg_len'])
    y = torch.ones(3,)

    with torch.autograd.set_detect_anomaly(True):
        outputs = net(x)
        
        recon_x = outputs['x_hat']
        # z = outputs['z']
        clz_proba = outputs['clz_proba']
        # clz_proba_voted = outputs['clz_proba_voted']
        clz_proba_age = outputs['clz_proba_age']
        
        # recon_x, z, proba_folds = net(x)
        print(
            f"recon_x:{recon_x.shape}, cls_proba:{clz_proba.shape}, "
            f"cls_proba_age:{clz_proba_age.shape},")
        
        criteria_classif = nn.CrossEntropyLoss()
        loss = criteria_classif(clz_proba, y.to(torch.int64))
        loss.backward()

if __name__ == '__main__':
    main()