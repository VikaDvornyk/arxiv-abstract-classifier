import unittest
from unittest.mock import MagicMock
from rest_framework.response import Response
from views import predict_abstract_category

class TestPredictAbstractCategory(unittest.TestCase):

    def test_predict_abstract_category(self):
        # create a mock request object
        mock_request = MagicMock()
        mock_request.data = {'abstract': 'This is an abstract.'}

        # call the function and capture the response
        response = predict_abstract_category(mock_request)

        # check that the response is a Response object
        self.assertIsInstance(response, Response)

        # check that the response contains the expected keys
        self.assertIn('category', response.data)
        self.assertIn('model_score', response.data)

        # check that the predicted category is one of the expected values
        expected_categories = set(['biology', 'chemistry', 'computer science', 'physics', 'social science'])
        predicted_categories = set(response.data['category'])
        self.assertTrue(predicted_categories.issubset(expected_categories))

        # check that the model scores are all floats between 0 and 1
        model_scores = response.data['model_score']
        self.assertIsInstance(model_scores, dict)
        for score in model_scores.values():
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

if __name__ == '__main__':
    unittest.main()


