# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math
import collections

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    # self.intial = None
    # self.count = None # Total count
    # self.sec = None
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  ############
  #computes and returns dict w probability of each value (normalized with total vals in list)
  def check(self, out):
    prob = dict(collections.Counter(out))
    for k in prob.keys():
      prob[k] = prob[k] / float(len(out))
    return prob
  ############
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"
    self.labelCount = [0]*len(self.legalLabels)
    self.featureCount = util.Counter()

    #creates empty dicts for each label in featureCount
    for x in self.legalLabels:
      self.featureCount[x] = util.Counter()
    
    for i in range(len(trainingData)):
      label = trainingLabels[i]
      self.labelCount[label] += 1 #incrementing count of labels in training data
      totalFeatures = trainingData[i]
      for y in totalFeatures:
        self.featureCount[label][y] += totalFeatures[y]  #counting 0s and 1s seen in feature y for label x
      
    self.totalTrainingData = len(trainingData)
    
    #use best k
    bestAcc = None
    #tempAcc = None
    bestK = self.k
    for k in kgrid or [0.0]:
      self.k = k
      classifiedRight = 0
      guessCount = self.classify(trainingData)
      for i, guess in enumerate(guessCount):
        if trainingLabels[i] == guess:
          classifiedRight += 1.0  #counts what is classified as right in the dataset to compare accuracies of k
      tempAcc = classifiedRight/len(guessCount) #percentage of accurate guesses

      if bestAcc is None or bestAcc < tempAcc:
        bestAcc = tempAcc
        bestK = k
    self.k = bestK

        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    
    "*** YOUR CODE HERE ***"
    self.priorProb = [1]*len(self.legalLabels)
    for i in range(len(self.labelCount)): #i is label
      if(self.labelCount[i] != 0):
        self.priorProb[i] = math.log(self.labelCount[i]/float(self.totalTrainingData))
        logJoint[i] = 1

        for featureData in datum:  #prob of being greater than 0 in this label OR less that 0
          if (datum[featureData] > 0):
            logJoint[i] += math.log((self.featureCount[i][featureData] + self.k)/(float(self.labelCount[i]) + self.k))
          else:
            logJoint[i] += math.log(((self.labelCount[i] - self.featureCount[i][featureData]) + self.k)/(float(self.labelCount[i]) + self.k))

        #stores prob into logJoint
        logJoint[i] = self.priorProb[i] +logJoint[i]
      else:
        continue
    return logJoint 

  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds
    

    
      
