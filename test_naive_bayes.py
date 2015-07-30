__author__ = 'sneha'
import naive_bayes

def test_summarize():
    data_set = [[1,20,0], [2,21,1], [3,22,0]]
    summary = naive_bayes.summarize(data_set)
    print('Attribute summaries: {0}').format(summary)
    attr1 = [1,2,3]
    attr2 = [20,21,22]
    assert summary == [(naive_bayes.mean(attr1), naive_bayes.stdev(attr1)), (naive_bayes.mean(attr2),
                                                                                naive_bayes.stdev(attr2))]


def test_summarize_by_class():
    dataset = [[1,20,1], [2,21,0], [3,22,1], [4,23,0]]
    summary = naive_bayes.summarize_by_class(dataset)
    assert summary[0] == naive_bayes.summarize([[2,21,0], [4,23,0]])
    assert summary[1] == naive_bayes.summarize([[1,20,1], [3,22,1]])


def test_calculate_class_probability():
    summaries = {0: [(1, 0.5)], 1:[(20, 5.0)]}

    inputVector = [1.1, '?']
    probabilties = naive_bayes.calculate_class_probabilities(summaries, inputVector)
    print probabilties
    probClass1 = naive_bayes.calculate_probability(1.1, 1, 0.5)
    probClass2 = naive_bayes.calculate_probability(1.1, 20, 5.0)
    assert probabilties[0] == probClass1
    assert probabilties[1] == probClass2

def test_predict():
    summaries = {'A': [(1, 0.5)], 'B': [(20, 5.0)]}
    inputVector = [1.1, '?']
    result = naive_bayes.predict(summaries, inputVector)
    probClassA = naive_bayes.calculate_probability(1.1, 1, 0.5)
    probClassB = naive_bayes.calculate_probability(1.1, 20, 5.0)
    prob = 'A' if probClassA > probClassB else 'B'
    assert prob == result

def test_get_predictions():
    summaries = {'A':[(1, 0.5)], 'B': [(20, 5.0)]}
    testSet = [[1.1, '?'], [19.1, '?']]
    predictions = naive_bayes.get_predictions(summaries, testSet)
    pred1 = naive_bayes.predict(summaries, testSet[0])
    pred2 = naive_bayes.predict(summaries, testSet[1])
    assert pred1 == predictions[0]
    assert pred2 == predictions[1]