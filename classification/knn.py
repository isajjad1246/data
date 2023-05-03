import util
import numpy as np 
import math
from math import sqrt

#get the euclidean distance between neighbors
#features are row vectors
def euclideanDistance(feature1, feature2):
  result = 0.0
  for i in range(len(feature1) - 1):
      result = result + (feature1[i] - feature2[i]) **2
  result = sqrt(result)
  # temp = feature1 - feature2
  # distance = np.sqrt(np.dot(temp.T, temp))
  return result

#finds neighbors with closer distances
def findNeighbors(train, testRow, n): #n is number of neighbors
  distances = list()  #(row index, distance) tuple
  for row in train:
    euclideanDist = euclideanDistance(testRow, row)
    distances.append((row, euclideanDist))
    
  
  #sort tuples based on distance lengths from shortest to longest
  distances.sort(key=lambda x: x[1])
  neighborList = list()
  for i in range(n):
    neighborList.append(distances[i][0])  #neighbor list has the neighbors in ascending distance order
  return neighborList

#find and return most common class label in k neighbors
def commonClassLabel(train, testRow, n):
  neighborList = findNeighbors(train, testRow, n)
  temp = []
  for row in neighborList:
    tempVal = row[-1]
    temp.append(tempVal)
  result = max(set(temp), key=temp.count)
  return result


#knn class- call in dataclassifier??
class KnnClassifier:

  #initialize method
  def __init__(self, legalLabels, neighbors):
    self.legalLabels = legalLabels
    self.type = "knn"
    self.n = neighbors
    
  #
  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    self.trainingData = trainingData
    self.trainingLabels = trainingLabels
    self.validationData = validationData
    self.validationLabels = validationLabels
    self.size = len(list(trainingData))

    #gets the features from each trainingData datum and puts in dict
    featureDict = []
    for datum in trainingData:
      feature = list(datum.values())
      featureDict.append(feature)

    #features and labels combined in training dict- this will be used for actual training in classify
    trainDict = []
    for i in range(self.size):
      trainingDatum = list(np.append(featureDict[i], self.trainingLabels[i])) #numpy append label to feature list
      trainDict.append(trainingDatum)
    self.trainDict = trainDict
    
  def classify(self, testData):
    self.size = len(list(testData))

    #same thing as in train except with the test data
    featureDict = []
    for datum in testData:
      feature = list(datum.values())
      featureDict.append(feature)
    testDict = []
    for i in range(self.size):
      trainingDatum = list(np.append(featureDict[i], self.trainingLabels[i]))
      testDict.append(trainingDatum)
    self.testDict = testDict
    

    #resultList holds predicted class labels for each test point
    resultList = []
    for datum in testDict:
      trainDict = self.trainDict
      n = self.n
      #here we just get the predicted label and put it in the list and then return
      result = commonClassLabel(trainDict, datum, n)
      resultList.append(result)
    return resultList
  
 