from transformers import AutoModel, AutoTokenizer

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import stanza
stanza.download('ru')
nlp = stanza.Pipeline('ru')

import pandas as pd

train_df = pd.read_csv("aug.csv")
group_theme_label2id = dict(
    (theme, idx) for (idx, theme) in enumerate(train_df["Группа тем"].unique().tolist())
)
group_theme_id2label = dict(
    (idx, theme) for (theme, idx) in group_theme_label2id.items()
)

theme_label2id = dict(
    (theme, idx) for (idx, theme) in enumerate(train_df["Тема"].unique().tolist())
)
theme_id2label = dict(
    (idx, theme) for (theme, idx) in theme_label2id.items()
)

class ThemeClassifier(nn.Module):

    def __init__(self, bert_model, freeze_bert=False):
        super(ThemeClassifier, self).__init__()

        # Размерность вектора предложения
        BERT_OUTPUT_DIM = 1024
        # Количество групп тем
        GROUP_THEME_NUM_CLASSES = 26
        # Количество тем
        THEME_NUM_CLASSES = 195

        # Размерность скрытого слоя
        HIDDEN_LAYER_1 = BERT_OUTPUT_DIM // 2

        # Коэффициент прореживания
        DROPOUT_RATE = 0.5

        self.bert_model = bert_model

        self.hidden = nn.Sequential(
            nn.Linear(BERT_OUTPUT_DIM, HIDDEN_LAYER_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        )

        self.group_theme = nn.Linear(HIDDEN_LAYER_1, GROUP_THEME_NUM_CLASSES)
        self.theme = nn.Linear(HIDDEN_LAYER_1, 256)

        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # Первый элемент выхода BERT-модели хранить все эмбеддинги токенов
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_model_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sentence_embedding = self.mean_pooling(bert_model_output, attention_mask)

        hidden_output = self.hidden(sentence_embedding)

        group_theme_logits = self.group_theme(hidden_output)
        theme_logits = self.theme(hidden_output)

        return group_theme_logits, theme_logits


class TokenizeInput:

    def __init__(self, pretrained_model_name_or_path, max_tokens=32):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.max_tokens = max_tokens

    def __call__(self, sentences):
        return self.tokenizer(
            sentences,
            padding='max_length',
            truncation=True,
            max_length=self.max_tokens,
            return_tensors='pt'
        )


class DvoraClassifier:

    def __init__(self):
        SBERT_MODEL_NAME = "ai-forever/sbert_large_nlu_ru"
        PRETRAINED_WEIGHTS = "pretrained_weights.pth"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = TokenizeInput(SBERT_MODEL_NAME)
        self.sbert = AutoModel.from_pretrained(SBERT_MODEL_NAME)
        self.classifier = ThemeClassifier(self.sbert, freeze_bert=True).to(self.device)

        if self.device == "cpu":
            self.classifier.load_state_dict(torch.load(PRETRAINED_WEIGHTS, map_location=torch.device('cpu')))
        else:
            self.classifier.load_state_dict(torch.load(PRETRAINED_WEIGHTS))
        self.classifier.eval()

    def __call__(self, sentences):
        encoded_input = self.tokenizer(sentences)

        ids = encoded_input["input_ids"].long().to(self.device)
        token_type_ids = encoded_input["token_type_ids"].long().to(self.device)
        attention_mask = encoded_input["attention_mask"].long().to(self.device)

        group_theme_logits, theme_logits = self.classifier(input_ids=ids, attention_mask=attention_mask,
                                                           token_type_ids=token_type_ids)

        group_theme_ids = group_theme_logits.argmax(dim=1).tolist()
        theme_ids = theme_logits.argmax(dim=1).tolist()

        result = []
        for (group_theme_id, theme_id) in zip(group_theme_ids, theme_ids):
            group_theme = group_theme_id2label[group_theme_id]
            if theme_id not in theme_id2label:
                theme = "Дороги"
            else:
                theme = theme_id2label[theme_id]
            result.append((group_theme, theme))

        return result


def _location(sentences):
    result = []

    if isinstance(sentences, str):
        sentences = [sentences]

    for sentence in sentences:
        found_locations = []
        doc = nlp(sentence)
        for el in doc.sentences:
            for ent in el.entities:
                if ent.type in 'LOC':
                    found_locations.append(ent.text)
    return result


_classifier = DvoraClassifier()


def dvora(sentences):
    themes = _classifier(sentences)
    locations = _location(sentences)

    info = []
    for i, th in enumerate(themes):
        loc = None
        if i < len(locations):
            loc = locations[i]
        info.append({"loc": loc, "group_theme": th[0], "theme": th[1]})

    return info