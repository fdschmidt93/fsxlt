from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_metric_learning import losses
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from transformers import AutoConfig
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.auto.modeling_auto import AutoModel
from trident import TridentModule

from .heads import ClassificationHead


@dataclass
class TokenClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    labels: Optional[torch.Tensor] = None


class AutoModelForCLSClassification(LightningModule):
    def __init__(self, hidden_dropout: float = 0.1, **kwargs):
        super().__init__()
        self.model = AutoModel.from_pretrained(**kwargs)
        # self.encoder = self.model  # alias
        self.dropout = nn.Dropout(hidden_dropout)
        self.num_labels = kwargs.get("num_labels", 2)
        self.classifier = ClassificationHead(
            hidden_size=self.model.config.hidden_size,
            num_labels=self.num_labels,
        )

    def forward(self, **kwargs) -> SequenceClassifierOutput:
        outputs = self.model(
            input_ids=kwargs.get("input_ids"),
            attention_mask=kwargs.get("attention_mask"),
            token_type_ids=kwargs.get("token_type_ids"),
            position_ids=kwargs.get("position_ids"),
            head_mask=kwargs.get("head_mask"),
            inputs_embeds=kwargs.get("inputs_embeds"),
            output_attentions=kwargs.get("output_attentions"),
            output_hidden_states=kwargs.get("output_hidden_states"),
            return_dict=kwargs.get("return_dict"),
        )
        sequence_output = outputs[0][:, 0, :]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        labels = kwargs.get("labels")
        if labels is not None:
            loss = cross_entropy(
                logits.view(-1, self.classifier.num_labels), labels.view(-1)
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AutoModelForTokenClassification(LightningModule):
    def __init__(self, hidden_dropout: float = 0.1, **kwargs):
        super().__init__()
        self.num_labels = kwargs.get("num_labels", 1)
        self.config = AutoConfig.from_pretrained(
            kwargs.get("pretrained_model_name_or_path")
        )
        self.config.update({"hidden_dropout_prob": hidden_dropout})
        self.roberta = AutoModel.from_pretrained(
            kwargs.get("pretrained_model_name_or_path"), config=self.config
        )
        self.hidden_size = self.roberta.config.hidden_size
        # self.encoder = self.model  # alias
        self.dropout = nn.Dropout(hidden_dropout)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def embed(self, *args):
        return self.roberta(input_ids=args[0], attention_mask=args[1])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> TokenClassifierOutput:
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            labels=labels,
        )


class OneVsAll(nn.Module):
    def __init__(self, num_labels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn((768, num_labels, 2)))

    def forward(self, x: torch.Tensor):
        # N, L, D
        logits = torch.einsum("nld, dcb->nlcb", x, self.weight)
        return logits


class AutoModelForTokenClassificationScratch(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_labels = kwargs.get("num_labels", 1)
        self.roberta = AutoModel.from_pretrained(**kwargs)
        self.hidden_size = self.roberta.config.hidden_size
        # self.encoder = self.model  # alias
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        # self.classifier = OneVsAll(self.num_labels)
        # self.classifier = nn.Linear(128, self.num_labels)

    # def one_hot_encoding(self, labels: torch.Tensor) -> torch.Tensor:
    #     N = labels.shape[0]
    #     one_hot = torch.zeros(
    #         (N, self.num_labels), dtype=torch.long, device=self.device
    #     )
    #     one_hot[torch.arange(N, device=self.device), labels] = 1
    #     return one_hot

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> TokenClassifierOutput:
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        sequence_output = outputs[0]
        # N, L, D = sequence_output.shape
        sequence_output = self.dropout(sequence_output)
        # w = self.classifier.weight
        # logits = sequence_output @ w.T  # + self.classifier.bias
        # nlcb
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # mask = torch.logical_and(attention_mask == 1, labels != -100)
            # # N * (L-pad), c, b)
            # loss_logits = logits[mask, :, :].view(-1, 2)
            # # N * (L-pad), C
            # one_hot = self.one_hot_encoding(labels[mask]).view(-1)
            # loss = F.cross_entropy(loss_logits, one_hot)
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        # probs = F.softmax(logits, -1)
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            # logits=probs[:, :, :, -1],
            hidden_states=outputs.hidden_states,
            labels=labels,
        )

    # def training_step(self, batch: dict, batch_idx: int) -> TokenClassifierOutput:
    #     attention_mask: torch.Tensor = batch["attention_mask"]
    #     labels: torch.Tensor = batch["labels"]
    #
    #     outputs = self.model.base_model(
    #         input_ids=batch["input_ids"],
    #         attention_mask=attention_mask,
    #     )
    #     sequence_output = outputs[0]
    #     N, L, D = sequence_output.shape
    #     sequence_output = sequence_output[labels >= 0]
    #     sequence_output = self.bn(sequence_output.view(-1, D)).view(N, L, D)
    #     sequence_output = self.model.dropout(sequence_output)
    #     logits = self.model.classifier(sequence_output)
    #
    #     loss = None
    #     if labels is not None:
    #         loss_fct = CrossEntropyLoss()
    #         # Only keep active parts of the loss
    #         if attention_mask is not None:
    #             active_loss = attention_mask.view(-1) == 1
    #             active_logits = logits.view(-1, self.num_labels)
    #             active_labels = torch.where(
    #                 active_loss,
    #                 labels.view(-1),
    #                 torch.tensor(loss_fct.ignore_index).type_as(labels),
    #             )
    #             loss = loss_fct(active_logits, active_labels)
    #         else:
    #             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    #
    #     return TokenClassifierOutput(
    #         loss=loss,
    #         logits=logits,
    #         hidden_states=outputs.hidden_states,
    #         attentions=outputs.attentions,
    #     )


class ContrastiveCLSTransformer(TridentModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = losses.SupConLoss(temperature=0.3)

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        outputs = self.model.model(
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            position_ids=batch.get("position_ids"),
            output_hidden_states=True,
        )
        sequence_output = outputs[0][:, 0, :]
        sequence_output = self.model.dropout(sequence_output)
        logits = self.model.classifier(sequence_output)
        clf_loss = None
        labels = batch["labels"]
        clf_loss = cross_entropy(
            logits.view(-1, self.model.classifier.num_labels), labels.view(-1)
        )
        scl = self.loss(sequence_output, labels)
        loss = scl * 0.9 + 0.1 * clf_loss
        self.log("train/loss", loss)
        return loss


class ContrastiveTokenTransformer(TridentModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = losses.SupConLoss(temperature=0.3)

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        outputs = self.model.base_model(
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            position_ids=batch.get("position_ids"),
            output_hidden_states=True,
        )
        labels = batch["labels"]
        ids = torch.where(labels != -100)
        labels = labels[ids]
        embeds = outputs["last_hidden_state"][ids]
        embeds = self.model.dropout(embeds)
        logits = self.model.classifier(embeds)
        clf_loss = None
        clf_loss = cross_entropy(
            logits.view(-1, self.trainer.datamodule.num_labels), labels.view(-1)
        )
        scl = self.loss(embeds, labels)
        loss = scl * 0.9 + 0.1 * clf_loss
        self.log("train/loss", loss)
        return loss
