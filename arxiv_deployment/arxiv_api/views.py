# Import necessary libraries
from rest_framework.decorators import api_view
from rest_framework.response import Response
import pandas as pd
from .apps import ArxivApiConfig
from src.custom_objects import score_text, preprocess_text

# Create your views here.
@api_view(['GET'])
def index(request):
    return_data = {
        "error_code": "0",
        "info": "success",
    }
    return Response(return_data)



@api_view(["POST"])
def predict_abstract_category(request):
    try:
        # load the request data
        article_json_info = request.data
        print(article_json_info['abstract'])

        # retrieve all the text from the json data
        text = article_json_info['abstract']
        # preprocess text
        text = preprocess_text(text)
        # predict category from trained model
        scores = score_text(text, ArxivApiConfig.model, ArxivApiConfig.tokenizer)[0]
        scores = pd.Series(scores, ['biology', 'chemistry', 'computer science', 'physics', 'social science'], name='scores')

        # when the second, third, and etc. score is > than 20%, it means multi-labels prediction
        scores_threshold = scores[scores > 0.2].sort_values(ascending=False)
        print(scores_threshold)

        model_prediction = {
            "category": scores_threshold.index,
            "model_score": scores_threshold
        }

    except ValueError as e:
        model_prediction = {
            'info': str(e)
        }

    return Response(model_prediction)