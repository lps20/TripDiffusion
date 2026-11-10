"""Conditional CTGAN / TVAE baselines (C-CTGAN, C-TVAE).

The demographic condition d (one-hot of the condition columns) is an *external* input:
  C-CTGAN: generator(z ⊕ c ⊕ d), discriminator(row ⊕ c ⊕ d); CTGAN's training-by-sampling
           conditional vector c covers the generated (trip) columns only.
  C-TVAE : encoder(x ⊕ d) -> z, decoder(z ⊕ d) -> x   (standard CVAE).
Only the trip columns are modelled; at sampling time one record is drawn per given
condition row (1:1 with the evaluation set).

Implementation note: these classes import the network modules, data transformer, data
sampler and loss of the installed `ctgan` package and add the external condition in the
training / sampling loops; no ctgan source is copied into this repository.
Training length is given in gradient steps (generator updates for C-CTGAN).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import optim

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.ctgan import CTGAN, Discriminator, Generator
from ctgan.synthesizers.tvae import Decoder, Encoder, _loss_function


def encode_condition(df: pd.DataFrame, cond_cols: Sequence[str], cond_sizes: Dict[str, int]) -> np.ndarray:
    parts = []
    for c in cond_cols:
        k = int(cond_sizes[c])
        v = df[c].to_numpy(dtype=np.int64).clip(0, k - 1)
        oh = np.zeros((len(df), k), dtype="float32")
        oh[np.arange(len(df)), v] = 1.0
        parts.append(oh)
    return np.concatenate(parts, axis=1)


def _infinite_epochs(n_steps: int, steps_per_epoch: int):
    done = 0
    while done < n_steps:
        for _ in range(min(steps_per_epoch, n_steps - done)):
            yield done
            done += 1


class ConditionalCTGAN(CTGAN):
    def __init__(self, cond_cols: Sequence[str], cond_sizes: Dict[str, int], steps: int, **kwargs):
        kwargs.setdefault("verbose", False)
        super().__init__(**kwargs)
        self.cond_cols = list(cond_cols)
        self.cond_sizes = dict(cond_sizes)
        self.steps = int(steps)
        self.log_every = 1000

    def fit_conditional(self, train_df: pd.DataFrame, gen_cols: Sequence[str]) -> None:
        gen_cols = list(gen_cols)
        self.gen_cols = gen_cols
        self._transformer = DataTransformer()
        self._transformer.fit(train_df[gen_cols], gen_cols)
        data = self._transformer.transform(train_df[gen_cols])
        D = encode_condition(train_df, self.cond_cols, self.cond_sizes)
        self._cond_dim = D.shape[1]
        self._data_sampler = DataSampler(data, self._transformer.output_info_list, self._log_frequency)
        data_dim = self._transformer.output_dimensions
        c_dim = self._data_sampler.dim_cond_vec()
        dev = self._device

        self._generator = Generator(self._embedding_dim + c_dim + self._cond_dim, self._generator_dim, data_dim).to(dev)
        disc = Discriminator(data_dim + c_dim + self._cond_dim, self._discriminator_dim, pac=self.pac).to(dev)
        opt_g = optim.Adam(self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
                           weight_decay=self._generator_decay)
        opt_d = optim.Adam(disc.parameters(), lr=self._discriminator_lr, betas=(0.5, 0.9),
                           weight_decay=self._discriminator_decay)
        B = self._batch_size
        rid = self._data_sampler._rid_by_cat_cols
        data_t = torch.from_numpy(data.astype("float32"))
        D_t = torch.from_numpy(D)

        def draw(col, opt):
            return np.array([np.random.choice(rid[c][o]) for c, o in zip(col, opt)])

        steps_per_epoch = max(len(data) // B, 1)
        logging.info("C-CTGAN: rows=%d data_dim=%d condvec=%d demo=%d steps=%d (%.1f epochs)",
                     len(data), data_dim, c_dim, self._cond_dim, self.steps, self.steps / steps_per_epoch)
        for step in _infinite_epochs(self.steps, steps_per_epoch):
            for _ in range(self._discriminator_steps):
                c1, m1, col, opt = self._data_sampler.sample_condvec(B)
                idx = draw(col, opt)                      # rows consistent with c1
                perm = np.random.permutation(B)
                c1 = torch.from_numpy(c1).to(dev)
                d_fake = D_t[idx].to(dev)                 # demographics of a real row with c1
                real = data_t[idx[perm]].to(dev)
                c2, d_real = c1[perm], D_t[idx[perm]].to(dev)
                z = torch.randn(B, self._embedding_dim, device=dev)
                fake = self._apply_activate(self._generator(torch.cat([z, c1, d_fake], 1)))
                fake_cat = torch.cat([fake, c1, d_fake], 1)
                real_cat = torch.cat([real, c2, d_real], 1)
                y_fake, y_real = disc(fake_cat), disc(real_cat)
                pen = disc.calc_gradient_penalty(real_cat, fake_cat, dev, self.pac)
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                opt_d.zero_grad(set_to_none=False)
                pen.backward(retain_graph=True)
                loss_d.backward()
                opt_d.step()

            c1, m1, col, opt = self._data_sampler.sample_condvec(B)
            idx = draw(col, opt)
            c1, m1 = torch.from_numpy(c1).to(dev), torch.from_numpy(m1).to(dev)
            d_fake = D_t[idx].to(dev)
            z = torch.randn(B, self._embedding_dim, device=dev)
            fake = self._generator(torch.cat([z, c1, d_fake], 1))
            y_fake = disc(torch.cat([self._apply_activate(fake), c1, d_fake], 1))
            loss_g = -torch.mean(y_fake) + self._cond_loss(fake, c1, m1)
            opt_g.zero_grad(set_to_none=False)
            loss_g.backward()
            opt_g.step()
            if (step + 1) % self.log_every == 0 or step + 1 == self.steps:
                logging.info("C-CTGAN step %d/%d loss_g=%.4f loss_d=%.4f", step + 1, self.steps,
                             float(loss_g), float(loss_d))

    @torch.no_grad()
    def sample_conditional(self, cond_df: pd.DataFrame) -> pd.DataFrame:
        self._generator.eval()
        D = encode_condition(cond_df, self.cond_cols, self.cond_sizes)
        out = []
        B = self._batch_size
        for s in range(0, len(D), B):
            d = torch.from_numpy(D[s:s + B]).to(self._device)
            n = d.size(0)
            c = torch.from_numpy(self._data_sampler.sample_original_condvec(n)).to(self._device)
            z = torch.randn(n, self._embedding_dim, device=self._device)
            out.append(self._apply_activate(self._generator(torch.cat([z, c, d], 1))).cpu().numpy())
        self._generator.train()
        gen = self._transformer.inverse_transform(np.concatenate(out, 0))
        gen = pd.DataFrame(gen, columns=self.gen_cols).reset_index(drop=True)
        for c in self.cond_cols:
            gen[c] = cond_df[c].to_numpy()
        return gen


class ConditionalTVAE:
    def __init__(self, cond_cols: Sequence[str], cond_sizes: Dict[str, int], steps: int,
                 embedding_dim: int = 128, compress_dims=(128, 128), decompress_dims=(128, 128),
                 l2scale: float = 1e-5, batch_size: int = 500, loss_factor: float = 2.0, lr: float = 1e-3,
                 device: Optional[str] = None):
        self.cond_cols = list(cond_cols)
        self.cond_sizes = dict(cond_sizes)
        self.steps = int(steps)
        self.embedding_dim = embedding_dim
        self.compress_dims = tuple(compress_dims)
        self.decompress_dims = tuple(decompress_dims)
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.lr = lr
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.log_every = 1000

    def fit_conditional(self, train_df: pd.DataFrame, gen_cols: Sequence[str]) -> None:
        self.gen_cols = list(gen_cols)
        self.transformer = DataTransformer()
        self.transformer.fit(train_df[self.gen_cols], self.gen_cols)
        X = torch.from_numpy(self.transformer.transform(train_df[self.gen_cols]).astype("float32"))
        D = torch.from_numpy(encode_condition(train_df, self.cond_cols, self.cond_sizes))
        data_dim, cdim = X.shape[1], D.shape[1]
        self.encoder = Encoder(data_dim + cdim, self.compress_dims, self.embedding_dim).to(self.device)
        self.decoder = Decoder(self.embedding_dim + cdim, self.decompress_dims, data_dim).to(self.device)
        opt = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                         lr=self.lr, weight_decay=self.l2scale)
        n = len(X)
        steps_per_epoch = max((n + self.batch_size - 1) // self.batch_size, 1)
        logging.info("C-TVAE: rows=%d data_dim=%d demo=%d steps=%d (%.1f epochs)",
                     n, data_dim, cdim, self.steps, self.steps / steps_per_epoch)
        perm = torch.randperm(n)
        pos = 0
        for step in range(self.steps):
            if pos >= n:
                perm, pos = torch.randperm(n), 0
            idx = perm[pos:pos + self.batch_size]
            pos += self.batch_size
            x, d = X[idx].to(self.device), D[idx].to(self.device)
            mu, std, logvar = self.encoder(torch.cat([x, d], 1))
            z = mu + torch.randn_like(std) * std
            rec, sigmas = self.decoder(torch.cat([z, d], 1))
            l1, l2 = _loss_function(rec, x, sigmas, mu, logvar, self.transformer.output_info_list, self.loss_factor)
            loss = l1 + l2
            opt.zero_grad()
            loss.backward()
            opt.step()
            self.decoder.sigma.data.clamp_(0.01, 1.0)
            if (step + 1) % self.log_every == 0 or step + 1 == self.steps:
                logging.info("C-TVAE step %d/%d loss=%.4f", step + 1, self.steps, float(loss))

    @torch.no_grad()
    def sample_conditional(self, cond_df: pd.DataFrame) -> pd.DataFrame:
        self.decoder.eval()
        D = torch.from_numpy(encode_condition(cond_df, self.cond_cols, self.cond_sizes))
        out, sig = [], None
        for s in range(0, len(D), self.batch_size):
            d = D[s:s + self.batch_size].to(self.device)
            z = torch.randn(d.size(0), self.embedding_dim, device=self.device)
            fake, sig = self.decoder(torch.cat([z, d], 1))
            out.append(torch.tanh(fake).cpu().numpy())
        self.decoder.train()
        gen = self.transformer.inverse_transform(np.concatenate(out, 0), sig.cpu().numpy())
        gen = pd.DataFrame(gen, columns=self.gen_cols).reset_index(drop=True)
        for c in self.cond_cols:
            gen[c] = cond_df[c].to_numpy()
        return gen
