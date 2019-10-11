# mira.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

# Mira implementation
import util
PRINT = False

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA. Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        bestCountCorrect = 0
        bestWeights = dict()

        for c in Cgrid:
            testWeights = self.weights.copy()

            for iteration in range(self.max_iterations):
                for i in range(len(trainingData)):
                    datum = trainingData[i]
                    correctLabel = trainingLabels[i]
                    scores = util.Counter()

                    for label in self.legalLabels:
                        # let the score of a given label be the dot product of the two vectors
                        scores[label] = testWeights[label] * datum

                    classificationLabel = scores.argMax()

                    if correctLabel != classificationLabel:
                        tau = ((testWeights[classificationLabel] - testWeights[correctLabel]) * datum + 1) / \
                              float(2 * (datum * datum))
                        tau = min(tau, c)

                        testDatum = datum.copy()
                        for f in testDatum:
                            testDatum[f] *= tau

                        # encourage correct label
                        testWeights[correctLabel] += testDatum
                        # discourage wrong label
                        testWeights[classificationLabel] -= testDatum

            # test accuracy of new classifier on validation data
            countCorrect = 0
            for i in range(len(validationData)):
                datum = validationData[i]
                correctLabel = validationLabels[i]
                scores = util.Counter()

                for label in self.legalLabels:
                    # let the score of a given label be the dot product of the two vectors
                    scores[label] = testWeights[label] * datum

                classificationLabel = scores.argMax()

                if classificationLabel == correctLabel:
                    countCorrect += 1
            countCorrect = countCorrect

            if countCorrect > bestCountCorrect:
                bestCountCorrect = countCorrect
                bestWeights = testWeights.copy()

        self.weights = bestWeights

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


    def findHighOddsFeatures(self, label1, label2):
        """
        Returns a list of the 100 features with the greatest difference in feature values
                         w_label1 - w_label2

        """
        return []
