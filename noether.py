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
        self.fc1 = nn.Linear(n_embd, n_embd)
        self.fc2 = nn.Linear(n_embd, 1)

    def forward(self, x, attention_mask=None):
        device = x.device

        # x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        range_tensor = torch.arange(x.size(1), device=device).unsqueeze(0)
        thought_idx = attention_mask.sum(-1)
        thought_mask = (range_tensor == thought_idx).long()
        print(thought_mask)
        assert False
        loss = torch.scatter_add(
            torch.zeros_like(x[:, :, 0], device=device),
            dim=-1,
            index=thought_mask,
            src=x[:, :, 0],
        )
        loss = loss[:, 1].mean(dim=0)

        return {"loss_ne": loss, "x_g": x}

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
        resume=False,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer  # torchopt optimizer
        self.inner_steps = inner_steps  # tailor steps
        self.inner_grad_clip = inner_grad_clip
        self.g_model = g_model
        if resume == False:
            self.thought_token = self.model.config.vocab_size
            self.update_model_embedding_table()
            print("NEW NOETHER MODEL: THOUGHT TOKEN = ", self.thought_token)
        else:
            self.thought_token = self.model.config.vocab_size - 1
            print("RESUMED NOETHER MODEL: THOUGHT TOKEN = ", self.thought_token)

    def update_model_embedding_table(self):

        gpt_weights = self.model.lm_head.weight
        thought_vector = gpt_weights.mean(dim=0)[None, :]
        new_weights = torch.cat((gpt_weights, thought_vector), dim=0)

        # update lm_head
        lm_head_new = nn.Linear(
            self.model.config.n_embd, self.model.config.vocab_size + 1, bias=False
        )
        lm_head_new.weight = nn.Parameter(new_weights)
        self.model.lm_head = lm_head_new
        # update embedding size
        self.model.transformer.wte = nn.Embedding(
            self.model.config.vocab_size + 1, self.model.config.n_embd
        )
        # update embedding weights
        self.model.transformer.wte.weight = self.model.lm_head.weight

    def tailor_forward(self, params, params_g, grad: bool, **kwargs):
        def loss(params, params_g, ids, targets, attention_mask, reduction, grad):
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

        print(kwargs["ids"])
        if grad:
            bs = kwargs["ids"].size(0)
            device = kwargs["ids"].device
            thought_idx = torch.stack(
                (
                    torch.arange(end=bs, device=device),
                    kwargs["attention_mask"].sum(-1),
                )
            ).long()
            # slot in thought token id
            kwargs["ids"] = torch.clone(kwargs["ids"])
            kwargs["ids"][thought_idx[0], thought_idx[1]] = self.thought_token
        print(kwargs["ids"])

        # grad is d_loss_ne/d_param
        grad_fn = torch.func.grad(loss, has_aux=True) if grad else loss
        grads, model_out = torch.vmap(grad_fn, in_dims=(0, None, 0, 0, 0, None, None))(
            params, params_g, *kwargs.values(), "mean", grad
        )
        model_out["logits"] = model_out["logits"][:, 0]
        return {
            "grads": grads if grad else None,
            "logits": model_out["logits"],
            "loss": loss,
            "loss_ne": model_out["loss_ne"],
            "x_g": model_out["x_g"],
            "x_pred": model_out["x_pred"],
        }

    def compute_masked_loss(self, logits, targets, mask, mask_val):
        bs = logits.size(0)
        seq_len = logits.size(1)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view(bs, seq_len)
        loss = loss * (mask == mask_val)
        loss = loss.sum(axis=1)
        num_loss_el = torch.sum(mask == mask_val, dim=-1)
        return loss, num_loss_el

    def forward(
        self,
        ids,
        targets=None,
        tailor_idx=None,
        target_mask=None,
        # reduction="mean",
    ):
        bs = ids.size(0)
        seq_len = ids.size(1)
        device = ids.device

        if tailor_idx == None:
            tailor_idx = torch.ones(bs) * (seq_len - 1)

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
        tailor_idx = tailor_idx.unsqueeze(1).to(device)
        attention_mask = (range_tensor < (tailor_idx - 1)).bool()

        # save original model logits
        original_logits = None

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
            # print(model_out["loss_ne"])
            # print(model_out["logits"])
            if i == 0:
                original_logits = model_out["logits"]

            params = torchopt.apply_updates(params, updates, inplace=False)

        # _, (logits, loss, loss_ne, _)
        model_out = self.tailor_forward(
            params,
            params_g,
            grad=False,
            **dict(ids=ids, targets=targets, attention_mask=attention_mask),
        )
        if type(target_mask) != type(None):
            # i think we want to compute loss only target_mask
            loss, num_loss_el = self.compute_masked_loss(
                model_out["logits"], targets, target_mask, mask_val=1
            )
            loss = loss / num_loss_el
        else:
            loss = torch.tensor(0.0)
        # if self.inner_steps > 0:
        #     loss_original, num_loss_el_original = self.compute_masked_loss(
        #         original_logits, targets, attention_mask, mask_val=1
        #     )
        #     loss += loss_original
        #     num_loss_el += num_loss_el_original

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
        inner_steps = self.inner_steps
        device = idx.device

        for _ in range(max_new_tokens):
            # if _ < 10:
            #     self.inner_steps = 0
            # else:
            #     self.inner_steps = inner_steps
            # if the sequence context is growing too long we must crop it at block_size
            # since we have a thought token ,we do block+size-1
            idx_cond = (
                idx
                if idx.size(1) <= self.model.config.block_size - 1
                else idx[:, -(self.model.config.block_size - 1) :]
            )
            # forward the model to get the logits for the index in the sequence
            # use dummmy targets
            idx_cond_with_thought_buffer = torch.cat(
                (idx_cond, torch.tensor([[0]], device=device).long()), dim=-1
            )
            model_out = self(idx_cond_with_thought_buffer, idx_cond_with_thought_buffer)
            logits = model_out["logits"]

            # pluck the logits at the final step and scale by desired temperature
            # since we are using a thought token buffer, we sample from the penultimate step
            logits = logits[:, -2, :] / temperature
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
