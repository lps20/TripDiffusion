import torch
import torch.nn as nn

from model.HCD_Net import TripDiffusionModel as BaseTripDiffusionModel


class TripDiffusionModel(BaseTripDiffusionModel):
    """
    Absorbing-state variant of the HCD model.
    - Categorical features use an extra MASK state as absorbing target.
    - Ordinal features keep the original Gaussian transition.
    """

    def __init__(self, features_info, cond_info, T, joint_pairs=None):
        super().__init__(features_info, cond_info, T, joint_pairs)
        self.mask_token_ids = {}
        self._expand_categorical_embeddings()
        self._rebuild_absorbing_diffusion_buffers()

    def _expand_categorical_embeddings(self):
        for feat in self.features_info:
            name = feat["name"]
            if feat["type"] != "categorical":
                self.mask_token_ids[name] = None
                continue

            old_emb = self.feature_embeddings[name]
            old_classes = old_emb.num_embeddings
            embed_dim = old_emb.embedding_dim

            new_emb = nn.Embedding(old_classes + 1, embed_dim)
            with torch.no_grad():
                new_emb.weight[:old_classes].copy_(old_emb.weight)
                new_emb.weight[old_classes].copy_(old_emb.weight.mean(dim=0))

            self.feature_embeddings[name] = new_emb
            self.mask_token_ids[name] = old_classes

    def _rebuild_absorbing_diffusion_buffers(self):
        # Remove diffusion-related buffers from the base class before re-registering.
        for feat in self.features_info:
            name = feat["name"]
            for prefix in ("trans", "cum_trans", "post"):
                attr_name = f"{prefix}_{name}"
                if hasattr(self, attr_name):
                    delattr(self, attr_name)

        for feat in self.features_info:
            name = feat["name"]
            orig_k = feat["num_classes"]
            feat_type = feat["type"]

            trans_list = []
            cum_list = []

            if feat_type == "categorical":
                noisy_k = orig_k + 1
                mask_idx = noisy_k - 1

                q_bar_prev = torch.zeros(orig_k, noisy_k)
                q_bar_prev[torch.arange(orig_k), torch.arange(orig_k)] = 1.0
                cum_list.append(q_bar_prev)

                for t in range(self.T):
                    beta_t = float(self.beta_schedule[t].item())
                    q_t = torch.zeros(noisy_k, noisy_k)
                    valid_idx = torch.arange(orig_k)

                    q_t[valid_idx, valid_idx] = 1.0 - beta_t
                    q_t[valid_idx, mask_idx] = beta_t
                    q_t[mask_idx, mask_idx] = 1.0

                    trans_list.append(q_t)
                    q_bar_prev = q_bar_prev @ q_t
                    cum_list.append(q_bar_prev)

            elif feat_type == "ordinal":
                q_bar_prev = torch.eye(orig_k)
                cum_list.append(q_bar_prev)

                for t in range(self.T):
                    sigma_t = float(self.sigma_schedule[t].item())
                    idx = torch.arange(orig_k).unsqueeze(1)
                    jdx = torch.arange(orig_k).unsqueeze(0)
                    dist_sq = (idx - jdx).float().pow(2)
                    q_t = torch.exp(-dist_sq / (2.0 * sigma_t ** 2))
                    q_t = q_t / q_t.sum(dim=1, keepdim=True)

                    trans_list.append(q_t)
                    q_bar_prev = q_bar_prev @ q_t
                    cum_list.append(q_bar_prev)

            else:
                raise ValueError(f"Unsupported feature type '{feat_type}' for feature '{name}'")

            # post_t maps p(x0|xt) -> p(x_{t-1}|xt) approximately through x0 mixing.
            # Shape is (T, num_x0_classes, num_noisy_states).
            post_list = [cum_list[t - 1] for t in range(1, self.T + 1)]

            self.register_buffer(f"trans_{name}", torch.stack(trans_list, dim=0))
            self.register_buffer(f"cum_trans_{name}", torch.stack(cum_list, dim=0))
            self.register_buffer(f"post_{name}", torch.stack(post_list, dim=0))
