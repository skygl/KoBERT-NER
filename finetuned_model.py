import torch
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertForTokenClassification, BertConfig


class BertNER(BertForTokenClassification):
    config_class = BertConfig

    def __init__(self, config, dataset_label_nums, multi_gpus=False):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dataset_label_nums = dataset_label_nums
        self.multi_gpus = multi_gpus
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifiers = torch.nn.ModuleList([torch.nn.Linear(config.hidden_size, x) for x in dataset_label_nums])
        self.background = torch.nn.Parameter(torch.zeros(1) - 2., requires_grad=True)

        self.init_weights()

    def forward(self, input_ids,
                attention_mask=None,
                dataset=0,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_logits=False):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifiers[dataset](sequence_output)
        outputs = torch.argmax(logits, dim=2)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.dataset_label_nums[dataset]), labels.view(-1))
            if output_logits:
                return loss, outputs, logits
            return loss, outputs
        else:
            if output_logits:
                return outputs, logits
            return outputs
