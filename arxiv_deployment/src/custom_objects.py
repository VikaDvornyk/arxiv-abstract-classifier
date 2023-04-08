import tensorflow as tf
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

MAX_LENGTH = 200

def multi_label_accuracy(y: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_pred = tf.math.round(y_pred)
    exact_matches = tf.math.reduce_all(y_pred == y, axis=1)
    exact_matches = tf.cast(exact_matches, tf.float32)
    return tf.math.reduce_mean(exact_matches)


def score_text(text, model, tokenizer):
    padded_encodings = tokenizer.encode_plus(
        text,
        max_length=MAX_LENGTH,  # truncates if len(s) > max_length
        return_token_type_ids=True,
        return_attention_mask=True,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )
    return model(padded_encodings["input_ids"]).numpy()

def preprocess_text(text):
    # clean the text
    text = re.sub('[^a-zA-Z]', ' ', text)
    # lowercase the words
    text = text.lower()
    # remove stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # perform lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text
