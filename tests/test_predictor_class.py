from .conf import predictor


def test_make_prediction(predictor):
    result = predictor[0].predict("hello there")
    assert isinstance(result, list)
    assert predictor[1] == len(result)


def test_make_prediction_anonymous(predictor):
    result = predictor[0].predict("hello there")
    assert isinstance(result, list)
    assert predictor[1] == len(result)


def test_getUsersCount(predictor):
    result = predictor[0].getUsersCount()
    assert result == predictor[1]


def test_getUsersList(predictor):
    result = predictor[0].getUsersList()
    assert result == predictor[2]
