from torch import nn, optim
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class HateSpeechClassifier(nn.Module):

    def __init__(self, model_name, n_classes):
        super(HateSpeechClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        output = self.drop(last_hidden_state[:, 0, :])
        return self.out(output)