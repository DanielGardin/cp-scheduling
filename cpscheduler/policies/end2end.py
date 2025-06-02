from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import layer_init, PositionalEncoding

class End2EndActor(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int = 1,
            n_layers: int = 1,
            dropout: float = 0.
        ):
        super().__init__()

        self.d_model = d_model

        self.embedding_fixed = nn.Embedding(2, 1)
        self.embedding_legal_op = nn.Embedding(2, 1)

        self.obs_projection = nn.Linear(4, d_model)
        layer_init(self.obs_projection)        

        self.pos_encoder = PositionalEncoding(d_model)

        transformer_encoder = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4*d_model, dropout=dropout, batch_first=True, norm_first=True
        )

        action_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4*d_model, dropout=dropout, batch_first=True, norm_first=True
        )

        self.job_encoder = nn.TransformerEncoder(
            transformer_encoder,
            n_layers,
            enable_nested_tensor=False
        )

        self.action_transformer = nn.TransformerEncoder(
            action_layer,
            n_layers,
            enable_nested_tensor=False
        )

        self.jobs_action = nn.Sequential(
            layer_init(nn.Linear(d_model, out_features=4*d_model)),
            nn.Tanh(),
            layer_init(nn.Linear(4*d_model, 1))
        )

        # self.no_op_action = nn.Sequential(
        #     layer_init(nn.Linear(d_model, 4*d_model)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(4*d_model, 1))
        # )


    def forward(self, obs: Tensor) -> Tensor:
        embedded_obs = torch.cat((
            self.embedding_fixed(obs[:, :, :, 0].long()),
            obs[:, :, :, 1:3],
            self.embedding_legal_op(obs[:, :, :, 3].long())),
        dim=3)

        batch_size, n_jobs, n_ops, n_features = embedded_obs.shape

        pos_encoding = self.pos_encoder(obs[:, :, :, -1].long())
        proj_obs = self.obs_projection(embedded_obs) + pos_encoding

        input_mask = obs[:, :, :, -1] == -1

        encoded_jobs = self.job_encoder(
            proj_obs.view(-1, n_ops, self.d_model),
            src_key_padding_mask=input_mask.view(-1, n_ops)
        ).view(batch_size, n_jobs, n_ops, self.d_model)

        encoded_jobs = encoded_jobs.mean(dim=2)

        machines     = obs[:, :, 1, 4]
        finished_job = input_mask[:, :, 1]

        job_resource_mask = ~(machines.unsqueeze(1) == machines.unsqueeze(-1))

        encodings = self.action_transformer(encoded_jobs, src_key_padding_mask=finished_job, mask=job_resource_mask)

        job_final = self.jobs_action(encodings)
        # no_op     = self.no_op_action(encodings)

        # logits = torch.cat((job_final.squeeze(2), no_op.mean(dim=1)), dim=1)

        # available_jobs = obs[:, :, 1, 3].bool()

        # no_op_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=obs.device)
        # mask = torch.cat((available_jobs, no_op_mask), dim=1)

        # logits = torch.masked_fill(logits, ~mask, -torch.inf)

        logits = job_final.squeeze(2)

        mask = torch.logical_or(
            finished_job,
            torch.logical_and(
                input_mask[:, :, 2],
                obs[:, :, 1, 0].bool()
            )
        )

        logits = torch.masked_fill(logits, mask, -torch.inf)

        return logits # type: ignore