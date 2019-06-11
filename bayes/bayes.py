from numpy import *

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0,1,0,1,0,1]
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet= set([])
    for document in dataSet:
        vocabSet = vocabSet | set (document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    rVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            rVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" %word
    return rVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float (numTrainDocs)
    p0Num = zeros(numWords) 
    p1Num = zeros(numWords) 
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range (numTrainDocs):
        if trainCategory[i] ==1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom    #change to log()
    p0Vect = p0Num / p0Denom    #change to log()
    return p0Vect, p1Vect, pAbusive

#printing the result
def printResult():
    loposts, lclasses = loadDataSet()
    myVocabList = createVocabList(loposts)
    trainMat=[]
    for postinDoc in loposts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, lclasses)
    print (p0V)
    print (p1V)
    print (pAb)

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log (pClass1)
    p0 = sum(vec2Classify * p0Vec) + log (1.0-pClass1)

    if p1>p0:
        return 1
    else: 
        return 0

#Testing NB
def go():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]

    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc =array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V,p1V, pAb) 
    
    testEntry = ['stupid', 'garbage']
    thisDoc =array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V,p1V, pAb)
