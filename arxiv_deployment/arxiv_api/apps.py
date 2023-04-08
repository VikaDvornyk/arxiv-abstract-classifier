from django.apps import AppConfig
from tensorflow.keras.models import load_model
from django.conf import settings
from arxiv_transformer_model.src.custom_objects import multi_label_accuracy
from transformers import BertTokenizerFast


class ArxivApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    model = load_model(f"{settings.BASE_DIR}/arxiv_transformer_model/fine_tuned_bert",
                       custom_objects={"multi_label_accuracy": multi_label_accuracy})
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    name = 'arxiv_api'
