from typing import Optional, Tuple

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses
from torch.distributions import Beta
from trident import TridentModule

from src.modules.functional import pooling
from src.projects.mixup.loss import (soft_cross_entropy,
                                     supervised_contrastive_loss)


def one_hot_encoding(y: torch.Tensor, num_labels: int = 3) -> torch.Tensor:
    N = y.shape[0]
    out = torch.zeros((N, num_labels), dtype=torch.long).to(y.device)
    out[torch.arange(N, device=y.device), y] = 1
    return out


class MixUpTransformer(TridentModule):
    def __init__(
        self,
        alpha: float,
        beta: Optional[float] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if beta is None:
            beta = alpha
        self.beta_dist = Beta(alpha, beta)

    def mixup(
        self,
        inst1: Tuple[torch.Tensor, torch.Tensor],
        inst2: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mixs up rows in x and y.

        Args:
            inst1 (:obj:`Tuple[torch.Tensor, torch.Tensor]`):
                Tuple of (x, y)-tensors for first instance
            inst2 (:obj:`Tuple[torch.Tensor, torch.Tensor]`):
                Tuple of (x, y)-tensors for second instance
        Returns:
            inputs, labels (:obj:`Tuple[torch.Tensor, torch.Tensor]`):
                Mixed-up inputs and labels

        Authors: Zhang et al.
        Affiliation: Facebook AI Research
        Original paper: https://arxiv.org/abs/1710.09412
        """
        x1, y1 = inst1
        x2, y2 = inst2

        N, C = y1.shape
        M = x2.shape[0]
        num = max(N, M)

        src_align = torch.randint(0, high=N, size=(num,), device=x1.device)
        trg_align = torch.randint(0, high=M, size=(num,), device=x1.device)

        x1 = x1[src_align]
        y1 = y1[src_align]
        x2 = x2[trg_align]
        y2 = y2[trg_align]

        lda = self.beta_dist.sample((num, 1)).to(x1.device)
        _lda = 1 - lda
        x_ = lda * x1 + _lda * x2
        y_ = lda * y1 + _lda * y2
        y_ = y_.view(num, C)
        return (x_, y_)

    def setup(self, stage: str):
        super().setup(stage)

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        # AutoModel.from_pretrained
        # DICT-style
        # >> loaders = {"x": loader_x, "y": loader_y, "z": loader_z}
        # >> {"x": batch_from_loader_x, "y": batch_from_loader_y, "z": batch_from_loader_z}
        # LIST-style
        # >> loaders = [loader_0, loader_1, loader_2]
        # >> [batch_from_loader_0, batch_from_loader_1, batch_from_loader_2]
        for b in batch.values():
            b_outputs = self.model.model(
                input_ids=b["input_ids"],
                attention_mask=b["attention_mask"],
                position_ids=b.get("position_ids", None),
                token_type_ids=b.get("token_type_ids", None),
            )
            b["cls"] = pooling.cls(b_outputs["last_hidden_state"])
            b["y"] = one_hot_encoding(b["labels"], self.model.num_labels)
        x_mixed, y_mixed = self.mixup(
            inst1=(batch["source"]["cls"], batch["source"]["y"]),
            inst2=(batch["target"]["cls"], batch["target"]["y"]),
        )
        x_mixed = self.model.dropout(x_mixed)
        logits = self.model.classifier(x_mixed)
        # x = torch.vstack([x_mixed, batches["target"]["cls"])
        # y = torch.vstack([y_mixed, batches["target"]["y"])
        loss = soft_cross_entropy(logits, y_mixed)
        self.log("train/loss", loss)
        return loss


class MultiTaskSequenceClassification(TridentModule):
    def __init__(self, reduce: str = "macro", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reduce = reduce

    def setup(self, stage: str):
        super().setup(stage)

    def pad_stack(self, x: torch.Tensor, y: torch.Tensor, value: int = -1):
        XN, XL = x.shape
        YN, YL = y.shape
        out = torch.full((XN + YN, max(XL, YL)), value, device=self.device)
        out[:XN, :XL] = x
        out[XN:, :YL] = y
        return out

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        if self.reduce == "macro":
            losses = {
                k: self.model(
                    input_ids=v["input_ids"],
                    attention_mask=v["attention_mask"],
                    labels=v["labels"],
                ).loss
                for k, v in batch.items()
            }
            for k, v in losses.items():
                self.log(f"{k}/loss", v)
            loss = torch.stack(list(losses.values())).mean()
        elif self.reduce == "micro":
            # 1 is xlm-r pad token id
            input_ids = self.pad_stack(
                batch["source"]["input_ids"], batch["target"]["input_ids"], 1
            )
            # 0 no attention for padding
            attention_mask = self.pad_stack(
                batch["source"]["attention_mask"], batch["target"]["attention_mask"], 0
            )
            # -100 ignore idx
            labels = self.pad_stack(
                batch["source"]["labels"], batch["target"]["labels"], -100
            )
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
        else:
            raise NotADirectoryError("Should be one of macro or micro")

        self.log("train/loss", loss)
        return loss


class ContrastiveMixUpTransformer(TridentModule):
    def __init__(self, alpha: float, beta: Optional[float] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if beta is None:
            beta = alpha
        self.beta_dist = Beta(alpha, beta)
        self.loss = losses.SupConLoss()

    def mixup(
        self,
        inst1: Tuple[torch.Tensor, torch.Tensor],
        inst2: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mixs up rows in x and y.

        Args:
            inst1 (:obj:`Tuple[torch.Tensor, torch.Tensor]`):
                Tuple of (x, y)-tensors for first instance
            inst2 (:obj:`Tuple[torch.Tensor, torch.Tensor]`):
                Tuple of (x, y)-tensors for second instance
        Returns:
            inputs, labels (:obj:`Tuple[torch.Tensor, torch.Tensor]`):
                Mixed-up inputs and labels

        Authors: Zhang et al.
        Affiliation: Facebook AI Research
        Original paper: https://arxiv.org/abs/1710.09412
        """
        x1, y1 = inst1
        x2, y2 = inst2

        N, C = y1.shape
        M = x2.shape[0]
        num = max(N, M)

        src_align = torch.randint(0, high=N, size=(num,), device=x1.device)
        trg_align = torch.randint(0, high=M, size=(num,), device=x1.device)

        x1 = x1[src_align]
        y1 = y1[src_align]
        x2 = x2[trg_align]
        y2 = y2[trg_align]

        lda = self.beta_dist.sample((num, 1)).to(x1.device)
        _lda = 1 - lda
        x_ = lda * x1 + _lda * x2
        y_ = lda * y1 + _lda * y2
        y_ = y_.view(num, C)
        return (x_, y_)

    def setup(self, stage: str):
        super().setup(stage)

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        # AutoModel.from_pretrained
        # DICT-style
        # >> loaders = {"x": loader_x, "y": loader_y, "z": loader_z}
        # >> {"x": batch_from_loader_x, "y": batch_from_loader_y, "z": batch_from_loader_z}
        # LIST-style
        # >> loaders = [loader_0, loader_1, loader_2]
        # >> [batch_from_loader_0, batch_from_loader_1, batch_from_loader_2]
        for b in batch.values():
            b_outputs = self.model.model(
                input_ids=b["input_ids"],
                attention_mask=b["attention_mask"],
                position_ids=b.get("position_ids", None),
                token_type_ids=b.get("token_type_ids", None),
            )
            b["cls"] = pooling.cls(b_outputs["last_hidden_state"])
            b["y"] = one_hot_encoding(b["labels"])
        x = torch.vstack([batch["source"]["cls"], batch["target"]["cls"]])
        y = torch.cat([batch["source"]["labels"], batch["target"]["labels"]])
        x = self.model.dropout(x)
        logits = self.model.classifier(x)
        loss = F.cross_entropy(logits, y)
        scl = self.loss(x, y)
        # x_mixed, y_mixed = self.mixup(
        #     inst1=(batch["source"]["cls"], batch["source"]["y"]),
        #     inst2=(batch["target"]["cls"], batch["target"]["y"]),
        # )
        # x_mixed = self.model.dropout(x_mixed)
        # logits = self.model.classifier(x_mixed)
        #
        # loss = soft_cross_entropy(logits, y_mixed)
        # scl = supervised_contrastive_loss(x_mixed, y_mixed)
        loss = scl * 0.9 + 0.1 * loss
        self.log("train/loss", loss)
        return loss


class MultiTaskTokenClassification(TridentModule):
    def __init__(
        self,
        reduce: str = "macro",
        gradient_checkpointing: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reduce = reduce
        self.gradient_checkpointing = gradient_checkpointing

    def setup(self, stage: str):
        super().setup(stage)
        if self.gradient_checkpointing:
            self.model.roberta.encoder.gradient_checkpointing = True
            print("Enabled checkpointing")

    def pad_stack(self, x: torch.Tensor, y: torch.Tensor, value: int = -1):
        XN, XL = x.shape
        YN, YL = y.shape
        out = torch.full((XN + YN, max(XL, YL)), value, device=self.device)
        out[:XN, :XL] = x
        out[XN:, :YL] = y
        return out

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        if self.reduce == "macro":
            losses = {
                k: self.model(
                    input_ids=v["input_ids"],
                    attention_mask=v["attention_mask"],
                    labels=v["labels"],
                ).loss
                for k, v in batch.items()
            }
            for k, v in losses.items():
                self.log(f"{k}/loss", v)
            loss = torch.stack(list(losses.values())).mean()
        elif self.reduce == "micro":
            # 1 is xlm-r pad token id
            input_ids = self.pad_stack(
                batch["source"]["input_ids"], batch["target"]["input_ids"], 1
            )
            # 0 no attention for padding
            attention_mask = self.pad_stack(
                batch["source"]["attention_mask"], batch["target"]["attention_mask"], 0
            )
            # -100 ignore idx
            labels = self.pad_stack(
                batch["source"]["labels"], batch["target"]["labels"], -100
            )
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            mask = labels != -100
            clf_loss = outputs.loss
            embeds = outputs.hidden_states[-1][mask]
            embeds = embeds / embeds.norm(2, -1, keepdim=True)
            scl_loss = self.loss(embeds, labels[mask])
            loss = clf_loss * 0.1 + scl_loss * 0.9
        else:
            raise NotADirectoryError("Should be one of macro or micro")

        self.log("train/loss", loss)
        return loss


class MixUpForTokenClassification(TridentModule):
    def __init__(
        self,
        alpha: float,
        beta: Optional[float] = None,
        gradient_checkpointing: bool = False,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        if beta is None:
            beta = alpha
        self.beta_dist = Beta(alpha, beta)
        self.gradient_checkpointing = gradient_checkpointing

    def mixup(
        self,
        inst1: Tuple[torch.Tensor, torch.Tensor],
        inst2: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mixs up rows in x and y.

        Args:
            inst1 (:obj:`Tuple[torch.Tensor, torch.Tensor]`):
                Tuple of (x, y)-tensors for first instance
            inst2 (:obj:`Tuple[torch.Tensor, torch.Tensor]`):
                Tuple of (x, y)-tensors for second instance
        Returns:
            inputs, labels (:obj:`Tuple[torch.Tensor, torch.Tensor]`):
                Mixed-up inputs and labels

        Authors: Zhang et al.
        Affiliation: Facebook AI Research
        Original paper: https://arxiv.org/abs/1710.09412
        """
        x1, y1 = inst1
        x2, y2 = inst2

        lda = self.beta_dist.sample(x1.shape[:1]).unsqueeze(-1).to(x1.device)
        _lda = 1 - lda

        x_ = lda * x1 + _lda * x2
        y_ = lda * y1 + _lda * y2

        return (x_, y_)

    def get_alignment(
        self,
        src_embeds: torch.Tensor,
        src_label: torch.Tensor,
        trg_embeds: torch.Tensor,
        trg_label: torch.Tensor,
    ):
        src_ids = torch.nonzero(
            src_label != -100, as_tuple=False
        )  # .chunk(chunks=2, dim=1)
        trg_ids = torch.nonzero(
            trg_label != -100, as_tuple=False
        )  # .chunk(chunks=2, dim=1)
        N = src_ids.shape[0]
        M = trg_ids.shape[0]
        num = max(N, M)

        src_align = torch.randint(0, high=N, size=(num,))
        trg_align = torch.randint(0, high=M, size=(num,))

        src_ids_aligned = src_ids[src_align].chunk(chunks=2, dim=1)
        trg_ids_aligned = trg_ids[trg_align].chunk(chunks=2, dim=1)

        src_aligned_labels = one_hot_encoding(
            src_label[src_ids_aligned].flatten(),
            num_labels=self.trainer.datamodule.num_labels,
        )
        trg_aligned_labels = one_hot_encoding(
            trg_label[trg_ids_aligned].flatten(),
            num_labels=self.trainer.datamodule.num_labels,
        )
        src_aligned_embeds = src_embeds[src_ids_aligned].squeeze(1)
        trg_aligned_embeds = trg_embeds[trg_ids_aligned].squeeze(1)
        return (src_aligned_embeds, src_aligned_labels), (
            trg_aligned_embeds,
            trg_aligned_labels,
        )

    def setup(self, stage: str):
        super().setup(stage)
        if self.gradient_checkpointing:
            self.model.roberta.encoder.gradient_checkpointing = True

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        for b in batch.values():
            b["last_hidden_state"] = self.model.base_model(
                b["input_ids"], b["attention_mask"]
            ).last_hidden_state
        src_embeds = batch["source"]["last_hidden_state"]
        src_labels = batch["source"]["labels"]
        trg_embeds = batch["target"]["last_hidden_state"]
        trg_labels = batch["target"]["labels"]

        src_inst, trg_inst = self.get_alignment(
            src_embeds=src_embeds,
            src_label=src_labels,
            trg_embeds=trg_embeds,
            trg_label=trg_labels,
        )
        x_mixed, y_mixed = self.mixup(
            inst1=src_inst,
            inst2=trg_inst,
        )
        x_mixed = self.model.dropout(x_mixed)

        logits = self.model.classifier(x_mixed)
        # logits = torch.utils.checkpoint.checkpoint_sequential([self.model.classifier], 4, x_mixed)

        loss = soft_cross_entropy(logits, y_mixed)
        self.log("train/loss", loss)
        return loss


class MixUpForMultipleChoice(TridentModule):
    def __init__(
        self,
        alpha: float,
        beta: Optional[float] = None,
        num_train_choices: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if beta is None:
            beta = alpha
        self.beta_dist = Beta(alpha, beta)
        self.num_train_choices = num_train_choices

    def mixup(
        self,
        inst1: Tuple[torch.Tensor, torch.Tensor],
        inst2: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mixs up rows in x and y.

        Args:
            inst1 (:obj:`Tuple[torch.Tensor, torch.Tensor]`):
                Tuple of (x, y)-tensors for first instance
            inst2 (:obj:`Tuple[torch.Tensor, torch.Tensor]`):
                Tuple of (x, y)-tensors for second instance
        Returns:
            inputs, labels (:obj:`Tuple[torch.Tensor, torch.Tensor]`):
                Mixed-up inputs and labels

        Authors: Zhang et al.
        Affiliation: Facebook AI Research
        Original paper: https://arxiv.org/abs/1710.09412
        """
        # BN, num_choices, emb_dim
        x1, y1 = inst1
        x2, y2 = inst2

        N = x1.shape[0]
        M = x2.shape[0]
        num = max(N, M)

        src_align = torch.randint(0, high=N, size=(num,))
        trg_align = torch.randint(0, high=M, size=(num,))

        x1 = x1[src_align]
        y1 = y1[src_align].view(num, 2, 1)
        x2 = x2[trg_align]
        y2 = y2[trg_align].view(num, 2, 1)

        lda = self.beta_dist.sample((num, 1, 1)).to(x1.device)
        _lda = 1 - lda
        x_ = lda * x1 + _lda * x2
        y_ = lda * y1 + _lda * y2
        y_ = y_.view(num, 2)
        return (x_, y_)

    def setup(self, stage: str):
        super().setup(stage)

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        for b in batch.values():
            # bs, num_choices, seq_len
            input_ids = b["input_ids"]
            N, num_choices, _ = input_ids.shape
            attention_mask = b["attention_mask"]
            flat_input_ids = (
                input_ids.view(-1, input_ids.size(-1))
                if input_ids is not None
                else None
            )
            flat_attention_mask = (
                attention_mask.view(-1, attention_mask.size(-1))
                if attention_mask is not None
                else None
            )
            b["cls"] = self.model.base_model(
                input_ids=flat_input_ids,
                attention_mask=flat_attention_mask,
            )[1].view(N, num_choices, -1)
            b["labels"] = one_hot_encoding(
                b["labels"], num_labels=self.num_train_choices
            )

        x_mixed, y_mixed = self.mixup(
            inst1=(batch["source"]["cls"], batch["source"]["labels"]),
            inst2=(batch["target"]["cls"], batch["target"]["labels"]),
        )
        x_mixed = self.model.dropout(x_mixed)
        logits = self.model.classifier(x_mixed).view(-1, 2)
        loss = soft_cross_entropy(logits, y_mixed)
        self.log("train/loss", loss)
        return loss


class MultiTaskMultipleChoice(TridentModule):
    def __init__(self, reduce: str = "macro", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reduce = reduce

    def setup(self, stage: str):
        super().setup(stage)

    def pad_stack(self, x: torch.Tensor, y: torch.Tensor, value: int = -1):
        XN, XL = x.shape
        YN, YL = y.shape
        out = torch.full((XN + YN, max(XL, YL)), value, device=self.device)
        out[:XN, :XL] = x
        out[XN:, :YL] = y
        return out

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        if self.reduce == "macro":
            losses = {
                k: self.model(
                    input_ids=v["input_ids"],
                    attention_mask=v["attention_mask"],
                    labels=v["labels"],
                ).loss
                for k, v in batch.items()
            }
            for k, v in losses.items():
                self.log(f"{k}/loss", v)
            loss = torch.stack(list(losses.values())).mean()
        elif self.reduce == "micro":
            # 1 is xlm-r pad token id
            input_ids = self.pad_stack(
                batch["source"]["input_ids"], batch["target"]["input_ids"], 1
            )
            # 0 no attention for padding
            attention_mask = self.pad_stack(
                batch["source"]["attention_mask"], batch["target"]["attention_mask"], 0
            )
            # -100 ignore idx
            labels = self.pad_stack(
                batch["source"]["labels"], batch["target"]["labels"], -100
            )
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
        else:
            raise NotADirectoryError("Should be one of macro or micro")

        self.log("train/loss", loss)
        return loss
