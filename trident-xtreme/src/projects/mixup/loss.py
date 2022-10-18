from typing import Optional

import torch
import torch.nn.functional as F


def soft_cross_entropy(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean-reduced soft cross-entropy."""
    logprobs = F.log_softmax(input, dim=-1)
    return -(target * logprobs).sum(dim=-1).mean()


# def supervised_contrastive_loss(
#     x: torch.Tensor,
#     labels: torch.Tensor,
#     tau: Optional[float] = 0.3,
#     normalize: bool = True,
# ) -> torch.Tensor:
#     # l2 normalisierung
#     if normalize:
#         x = x / x.norm(p=2, dim=-1, keepdim=True)
#
#     # outer product: pair-wise inner product; pair-wise cosine similarities
#     scores = x @ x.T
#
#     # tau: temperature
#     if tau is not None:
#         scores = scores / tau
#
#     labels_ = labels[:, None] == labels[None]
#
#     # mask out diagnoal to avoid self-labels
#     # TODO think about SimCSE
#     scores.fill_diagonal_(-100000)
#     labels_.fill_diagonal_(False)
#
#     # efficient max trick -- any constant suffices
#     max_ = scores.max(1, keepdim=True)[0].detach()
#     scores = scores - max_
#     scores_exp = torch.exp(scores)
#
#     # avoid log(0)
#     # denom = scores_exp + (scores_exp * torch.logical_not(labels_)).sum(1) denom = torch.log(denom.clamp(1e-32))
#     denom = scores_exp.sum(1).log().clamp(1e-16)
#
#     log_loss = (-scores + denom) * labels_
#
#     # avoid nan from dividing by zero
#     avg = labels_.sum(1)
#     loss = log_loss.sum(1) / avg.clamp(1)
#     loss = loss.masked_select(avg.bool()).mean()  # / 2
#     return loss
#
#
# import pytorch_metric_learning.utils.common_functions as c_f
# import pytorch_metric_learning.utils.loss_and_miner_utils as lmu
# from pytorch_metric_learning import losses
#
# l = losses.SupConLoss()
#
#
# pos_mask = labels_
# neg_mask = ~labels_
# mat = scores
# mat_max, _ = mat.max(dim=1, keepdim=True)
# mat = mat - mat_max.detach()  # for numerical stability
# denominator = lmu.logsumexp(
#     mat, keep_mask=(pos_mask + neg_mask).bool(), add_one=False, dim=1
# )
# log_prob = mat - denominator
# mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (
#     pos_mask.sum(dim=1) + c_f.small_val(mat.dtype)
# )
#
#
# loss_ = l(x, labels)
#

def supervised_contrastive_loss(
    x: torch.Tensor,
    labels: torch.Tensor,
    tau: Optional[float] = 0.3,
    normalize: bool = True,
) -> torch.Tensor:
    # import pytorch_metric_learning.utils.common_functions as c_f
    # import pytorch_metric_learning.utils.loss_and_miner_utils as lmu
    # from pytorch_metric_learning import losses
    #
    # l = losses.SupConLoss()
    #
    # N = 30
    # C = 3
    # x = torch.rand((N, 768))
    # x = x / x.norm(p=2, dim=-1, keepdim=True)
    # labels = torch.randint(low=0, high=C, size=(N,))
    # loss_ = l(x, labels)
    # labels = F.one_hot(labels)

    # labels = torch.rand((N, 3))
    # labels /= labels.sum(1, keepdim=True)

    N = x.shape[0]
    # if normalize:
    #     x = x / x.norm(p=2, dim=-1, keepdim=True)

    # outer product: pair-wise inner product; pair-wise cosine similarities
    x = x / x.norm(p=2, dim=-1, keepdim=True)
    scores = x @ x.T / tau

    # tau: temperature
    # if tau is not None:
    #     scores = scores / tau

    indicator = labels > 0.0
    loss_weight = labels[:, None, :] * labels[None, :, :]
    # mask out diagnoal to avoid self-labels
    diagonal_mask = (
        torch.full((N, N), fill_value=1, device=x.device)
        .fill_diagonal_(0)
        # .unsqueeze(-1)
    )
    # indicator = indicator.unsqueeze(1) * diagonal_mask
    loss_weight *= diagonal_mask.unsqueeze(-1)

    # TODO think about SimCSE
    scores.fill_diagonal_(-60000)

    # efficient max trick -- any constant suffices
    max_ = scores.max(dim=1, keepdim=True)[0].detach()
    scores = scores - max_
    scores_exp = torch.exp(scores).unsqueeze(-1)

    denom = scores_exp.sum(dim=1).log().clamp(1e-16)

    log_loss = (((scores.unsqueeze(-1) - denom) * loss_weight)).sum(
        1
    ) / loss_weight.sum(1)
    loss = -log_loss.masked_select(indicator).mean()
    return loss
