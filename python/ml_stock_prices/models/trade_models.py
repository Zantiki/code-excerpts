from abc import abstractmethod
from datetime import datetime
from src.models.models import RandomForest


class Evaluator:

    def __init__(self, classification_model=RandomForest(100)):
        # needs to be pre-initialized, is a torch-object
        self.model_object = classification_model
        self.last_updated = datetime.now()

    @abstractmethod
    def evaluate(self, history):
        pass

    def train(self, input, expected_out):
        final_accuracy = self.model_object.train(input, expected_out)
        return final_accuracy


class StockEvaluator(Evaluator):

    """
    Give a descrete score to a stock based upon probable up/down movement. Classification, start with random forest
    """
    def __init__(self, model_object=RandomForest(100)):
        super().__init__(model_object)

    def evaluate(self, history):
        pass


class PriceEvaluator(Evaluator):
    """
    Provide an entry and exit based on prior data and stock-evaluator score. Regression, start with a GRU
    """
    def __init__(self, model_object=RandomForest(100)):
        super().__init__(model_object)

    def evaluate(self, history):
        """
        returns expected entry and exit calculated using torch
        :param history: the history of a stock in a given time-range
        :return: expected entry and exit
        """
        # do something with history
        pass


