import util
import numpy as np 
import math
from math import sqrt

#get the euclidean distance between neighbors
def euclideanDistance(feature1, feature2):
    # distance = 0.0
    # for i in range(len(feature1) - 1):
    #     distance += (feature1[i] - feature2[i]) **2
    #return sqrt(distance)
    temp = feature1 - feature2
    distance = np.sqrt(np.dot(temp.T, temp))
    return distance

#finds neighbors with closer distances
def findNeighbors(train, testRow, numNeighbors):
    distances = list()
    for row in train:
        euclideanDist = euclideanDistance(testRow, row)
        distances.append((row, euclideanDist))
    
    #sort distance lengths from shortest to longest- change code for this
    distances.sort(key=lambda tup: tup[1])
    neighborList = list()
    for i in range(numNeighbors):
        neighborList.append(distances[i][0])
    return neighborList

#find and return most common class label in k neighbors
def commonClassLabel(train, testRow, numNeighbors):
    neighborList = findNeighbors(train, testRow, numNeighbors)
    temp = [row[-1] for row in neighborList]
    result = max(set(temp), key=temp.count)
    return result

class KnnClassifier:
    def __init__(self, legalLabels, neighbors):
        self.legalLabels = legalLabels
        self.type = "knn"
        self.numNeighbors = neighbors
    
    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        self.trainingData = trainingData
		self.trainingLabels = trainingLabels
		self.validationData = validationData
		self.validationLabels = validationLabels
        self.size = len(list(trainingData))

        featureDict = []
        for datum in trainingData:
            feature = list(datum.values())
            featureDict.append(feature)

        trainDict = []
        for i in range(self.size):
            trainingDatum = list(np.append(featureDict[i], self.trainingLabels[i]))
            trainDict.append(trainingDatum)
        self.trainDict = trainDict
    
    def classify(self, testData):
        self.size = len(list(testData))
        featureDict = []
        for datum in trainingData:
            feature = list(datum.values())
            featureDict.append(feature)
        
        testDict = []
        for i in range(self.size):
            trainingDatum = list(np.append(featureDict[i], self.trainingLabels[i]))
            testDict.append(trainingDatum)
        self.testDict = testDict
    
        resultList = []
        for datum in testDict:
            trainDict = self.trainDict
            numNeighbors = self.numNeighbors
            result = commonClassLabel(trainDict, datum, numNeighbors)
            resultList.append(result)
        return resultList
