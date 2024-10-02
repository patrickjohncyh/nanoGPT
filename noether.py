import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchopt


class MLPNoether(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros((1, n_embd)))

    def forward(self, x, attention_mask=None):
        x = x * F.sigmoid(self.gate)

        return self.conservation_loss(x, attention_mask)

    def conservation_loss(self, x, attention_mask):
        device = x.device
        bs = x.size(0)
        seq_len = x.size(-2)

        if seq_len == 1:
            return {"loss_ne": torch.tensor(0, device=device), "x_g": x}

        x_a = x[:, :-1, :]
        x_b = x[:, 1:, :]

        attention_mask = attention_mask[None, 1:]

        indexing_mask = (attention_mask == 0).long()
        loss = ((x_a - x_b) ** 2).sum(-1)  # loss per batch per element
        # print(x)
        # print(loss)
        # print(loss.shape)

        loss = torch.scatter_add(
            torch.zeros_like(loss, device=device),
            dim=-1,
            index=indexing_mask,
            src=loss,
        )
        # div by seq len
        loss = loss / (torch.sum(attention_mask, dim=-1) + 1)

        loss = loss[:, 0].mean(dim=0)

        return {"loss_ne": loss, "x_g": x}


class Noether(nn.Module):
    def __init__(
        self,
        model,
        optimizer,
        g_model,
        inner_steps: int = 1,
        inner_grad_clip: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer  # torchopt optimizer
        self.inner_steps = inner_steps  # tailor steps
        self.inner_grad_clip = inner_grad_clip
        self.g_model = g_model

    def tailor_forward(self, params, params_g, grad: bool, **kwargs):
        def loss(params, params_g, ids, targets, attention_mask, reduction):
            # make prediction on input sequence first
            # logits, loss, x_pred
            model_out = torch.func.functional_call(
                self.model,
                params,
                (
                    ids[None, ...],
                    targets[None, ...],
                    # reduction,
                ),
            )

            # pass seq x into model_g, and compute conservation loss
            # loss_ne, x_g
            model_g_out = torch.func.functional_call(
                self.g_model, params_g, (model_out["x_pred"], attention_mask)
            )

            return model_g_out["loss_ne"], {**model_out, **model_g_out}

        # grad is d_loss_ne/d_param
        grad = torch.func.grad(loss, has_aux=True) if grad else loss
        grads, model_out = torch.vmap(grad, in_dims=(0, None, 0, 0, 0, None))(
            params, params_g, *kwargs.values(), "mean"
        )
        return {
            "grads": grads if grad else None,
            "logits": model_out["logits"],
            "loss": loss,
            "loss_ne": model_out["loss_ne"],
            "x_g": model_out["x_g"],
            "x_pred": model_out["x_pred"],
        }

    def forward(
        self,
        ids,
        targets=None,
        tailor_idx=None,
        # reduction="mean",
    ):
        bs = ids.size(0)
        seq_len = ids.size(1)
        device = ids.device
        if tailor_idx == None:
            tailor_idx = torch.ones(bs) * seq_len

        invalid = [
            # "wte",
            # "wpe",
        ]
        # get model params and repeat bs times
        params = {
            k: v[None, ...].repeat((bs,) + (1,) * len(v.shape))
            for k, v in self.model.named_parameters()
            if not any(_ in k for _ in invalid)
        }
        params_g = dict(self.g_model.named_parameters())

        # Initial State
        state = self.optimizer.init(params)  # initial state

        # update model(s) for with N tailoring loss steps
        inner_step_grad_norm = 0

        # convert tailor_idx to attention_mask
        range_tensor = torch.arange(ids.size(1), device=device).unsqueeze(0)
        tailor_idx = tailor_idx.unsqueeze(1)
        attention_mask = (range_tensor < tailor_idx).bool()

        for i in range(self.inner_steps):
            # print(i)
            # grad is d_ether_loss/d_param
            # grads, (_, _, _, _)
            model_out = self.tailor_forward(
                params,
                params_g,
                grad=True,
                **dict(ids=ids, targets=targets, attention_mask=attention_mask),
            )
            # grads = {k: torch.nan_to_num(v, 0.0) for k, v in model_out["grads"].items()}
            updates, state = self.optimizer.update(
                model_out["grads"], state, inplace=False
            )

            params = torchopt.apply_updates(params, updates, inplace=False)

        # make prediction on updated model
        # _, (logits, loss, loss_ne, _)
        model_out = self.tailor_forward(
            params,
            params_g,
            grad=False,
            **dict(ids=ids, targets=targets, attention_mask=attention_mask),
        )

        # i think we want to compute loss only non-tailored inputs; i.e., where attention_mask==0
        # hence, recompute loss manually here : - (
        loss = F.cross_entropy(
            model_out["logits"].view(-1, model_out["logits"].size(-1)),
            targets.view(-1),
            reduction="none",
        ).view(bs, seq_len, -1)
        loss = loss[:, :, 0]
        loss = loss * (attention_mask == 0)
        loss = loss.sum(axis=1) / torch.sum(attention_mask == 0, dim=-1)
        # print(loss)
        # print(tailor_idx)
        # print("===")
        # loss = loss[
        #     batch_idx, pred_idx
        # ]  # extract only the loss from the last time f step
        # # usually we return logits, loss but for meta model we only care about loss?
        return {
            "logits": model_out["logits"].squeeze(1),
            "loss": torch.mean(loss),
            "loss_ne": model_out["loss_ne"],
            "x_g": model_out["x_g"],
            "x_pred": model_out["x_pred"],
        }

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.model.config.block_size
                else idx[:, -self.model.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            # use dummmy targets
            model_out = self(idx_cond, idx_cond)
            logits = model_out["logits"]
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
