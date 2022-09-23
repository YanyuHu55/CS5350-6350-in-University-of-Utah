from math import log
import numpy as np
import operator

# create the training data set
def createDataset():
    dataSet = np.loadtxt('car_train.csv', dtype=str, delimiter=",")
    attributes = ['buying', 'maint', 'doors', 'persons','lug_boot','safety']
    # print(dataset)
    # print(attributes)
    return dataSet, attributes

# calculate the Shannon Entropy
def shannonEnt(dataSet):
    # the number of training data
    len_dataSet = len(dataSet)
    # create a dictionary, calculate thetimes of each attributes
    attributeCounts = {}
    shannonEnt = 0.0 # Initial condition of Shannon Entropy = 0

    for element in dataSet:# analysis each data
        currentAttri = element[-1] # Extract attribute value information
        if currentAttri not in attributeCounts.keys(): # put the attribute name in the dic as key
            attributeCounts[currentAttri] = 0 # Initial condition of each attribute = 0
        attributeCounts[currentAttri] += 1 # count the number of the occurrence of attribute
    for key in attributeCounts: # calculate the Shannon Entropy of attribute
        proportion = float(attributeCounts[key])/len_dataSet
        shannonEnt -= proportion*log(proportion, 2)
    # print('the accumulation of attribute:{}'.format(attributeCounts))
    # print('Shannon Etropy:{}'.format(shannonEnt))
    return shannonEnt


def GiniIndex(dataSet):
    # the number of training data
    len_dataSet = len(dataSet)
    # create a dictionary, calculate thetimes of each attributes
    attributeCounts = {}
    GiniIndex = 1.0 # Initial condition of Gini Index = 0

    for element in dataSet:# analysis each data
        currentAttri = element[-1] # Extract attribute value information
        if currentAttri not in attributeCounts.keys(): # put the attribute name in the dic as key
            attributeCounts[currentAttri] = 0 # Initial condition of each attribute = 0
        attributeCounts[currentAttri] += 1 # count the number of the occurrence of attribute
    for key in attributeCounts: # calculate the Gini Index of attribute
        proportion = float(attributeCounts[key])/len_dataSet
        GiniIndex = GiniIndex - proportion*proportion
    # print('the accumulation of attribute:{}'.format(attributeCounts))
    # print('Gini Index:{}'.format(GiniIndex))
    return GiniIndex


def MajorError(dataSet):
    # the number of training data
    len_dataSet = len(dataSet)
    # create a dictionary, calculate thetimes of each attributes
    attributeCounts = {}
    ErrorElement = len_dataSet # Initial condition of Major Error

    for element in dataSet:# analysis each data
        currentAttri = element[-1] # Extract attribute value information
        if currentAttri not in attributeCounts.keys(): # put the attribute name in the dic as key
            attributeCounts[currentAttri] = 0 # Initial condition of each attribute = 0
        attributeCounts[currentAttri] += 1 # count the number of the occurrence of attribute
    for key in attributeCounts: # calculate the Major Error of attribute
        if key <= ErrorElement:
            ErrorElement = key

    MajorError= ErrorElement/len_dataSet
    # print('the accumulation of attribute:{}'.format(attributeCounts))
    # print('MajorError:{}'.format(MajorError))
    return MajorError




# split the dataset to calculate the entropy under some condition
# i: the index (from 0) of the attribute that use to split the dataset
# value the value that need to be returned
def splitDataSet(dataSet,i,value):
    # print(value, '\n')
    splitDataSet = [] # Store the partitioned dataset
    for example in dataSet:
        if example[i] == value:
            splitExample = example[:i]
            a=np.array(splitExample)
            b=np.array(example[i+1:])
            #a_list=list(a)
            #b_list=list(b)
            #a_list.extend(b_list )
            splitExample = np.concatenate((a,b),axis=0)
            #splitExample.extend(example[i+1:])
            splitDataSet.append(splitExample)
    return splitDataSet

# calculate the infoarmation gain and choose the best feature
def BestFeature(dataSet):
    print('bestfeatures dataset lengtth', len(dataSet),'best feature is', dataSet, )
    numFeature = len(dataSet[0]) -1  # the number of attributes
    baseEntropy=shannonEnt(dataSet) # the Shannon Entropy of the training dataset
    # print(baseEntropy)
    bestInfoGain = 0.0  # Initial condition of Information gain = 0
    bestFeature = -1  # the index of best feature in the attribute set
    # print(numFeature)
    gain = []
    Attribute = []
    for i in range(numFeature):
        featList = [example[i] for example in dataSet]# get the ith attribute of all dataset
        print('feat list', featList)
        print('featlist len=', len(featList), 'featlist type', type(featList))
        # print('featvalue=', featValue)
        # featValue = set(np.array(featList))# create a set
        featValue = []
        [featValue.append(i) for i in featList if i not in featValue]
        print(featValue)
        attributes = list(featValue)
        # print(featList)
        newEntropy = 0.0  # Initial condition of Entropy = 0
        for value in featValue:  # be focus
            subDataSet = splitDataSet(dataSet, i, value)  # call the function to partition the dataset
            proportion = float(len(subDataSet)/len(dataSet))
            # print(i, featList.index(value), proportion)
            # print(featList.index(value), value, subDataSet, '\n')
            newEntropy +=proportion*shannonEnt(subDataSet)  # calculate the conditional Entropy
            # newEntropy +=proportion*GiniIndex(subDataSet)  # calculate the conditional Gini Index
            # newEntropy +=proportion*MajorError(subDataSet)  # calculate the conditional Majority Error
        # print(i, newEntropy, '\n')
        infoGain = baseEntropy-newEntropy  # Information Gain
        # print(i, attributes, '\n')
        # print('The information gain of attribute %sis %.3f' % (attributes[i], infoGain))
        # if infoGain > bestInfoGain: # choose the best feature and it's index
        #     bestInfoGain = infoGain
        #     bestFeature = i
        gain.append(infoGain)
        Attribute.append(attributes)
    # print(gain)
    # print(Attribute)
    # print(infoGain, bestInfoGain, attributes)
    print('The best attribute is: %s' % Attribute[gain.index(max(gain))])
    return gain.index(max(gain)), Attribute[gain.index(max(gain))]



# get the most attribute in Classlist
# def maxAttribute(classList):
#     classCount = {}
#     for vote in classList: # count the times of element
#         if vote not in classCount.keys():
#             classCount[vote] = 0
#         classCount += 1
#     sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
#     return sortedClassCount[0][0]


# decision tree
# def creatTree(dataSet, attributes, featAttributes):
#     classList = [example[-1] for example in dataSet]  # get the ith attribute of all dataset
#     if classList. count(classList[0]==len(classList)): # Stop If the attribute is same
#         return classList[0]
#     if len(dataSet[0])==1:
#         return maxAttribute(classList)
#     bestFeature = BestFeature(dataSet)
#     bestFeatureAtrri = attributes[bestFeature]
#     featAttributes.append(bestFeatureAtrri)
#
#     myTree = {bestFeatureAtrri:{}}
#     del(attributes[bestFeature])
#     featureList = [example[bestFeature] for example in dataSet]
#     featureValue = set(featureList)
#     for value in featureValue:
#         subattributes = attributes[:]
#         myTree[bestFeatureAtrri][value] = creatTree(splitDataSet(dataSet, bestFeature,value), subattributes, featAttributes)
#         print(featAttributes)
#         return myTree

# def gini(data_list):
# createDataset()
# data_return, attri_return=createDataset()
# print(data_return)
# shannonEnt(data_return)
# BestFeature(data_return)
# featattribute = []

#####################################  first layer   ################################
train_data, attributes = createDataset()

depth = 1  #
input_depth = 4#

IndexBestFeat, bestFeature = BestFeature(train_data)
print('bestFeature:', bestFeature)
print(IndexBestFeat)
final_Label = {}
unitTrainData=[]
unitTrainData.append(train_data)

# treeLoop(train_data, bestFeature, final_Label, input_depth, depth)
# for i in range(depth):
#     index = BestFeature(train_data)
#     print(index)

# creatTree(data_return, attri_return, featattribute)

#bestFeature1 = ['low', 'med', 'high']
#print('bestFeature2', bestFeature1)
#ClassifyIndex = len(bestFeature)

##############################################################loop#############

# def buildTree(depth, input_depth, IndexBestFeat, bestFeature, train_data, attributes, final_Label):
#     for subUnitTrain in train_data:
#         # the length of dataset and the dataset after split depend on the best feat
#         dataclassifySet = [[] for _ in range(len(bestFeature[train_data.index(subUnitTrain)]))]
#         # IndexBestFeat = 5
#         for BestSingleFeat in bestFeature:  # sample_data in train_data:
#             for sample_data in train_data:
#                 if sample_data[IndexBestFeat[train_data.index(subUnitTrain)]] == BestSingleFeat:
#                     dataclassifySet[bestFeature.index(BestSingleFeat)].append(list(sample_data))
#         print('len(dataclassifySet)= ', len(dataclassifySet), 'dataclassifySet =', dataclassifySet, '\n')
#
#         # extract the the label from the dataset that is split based on the best feture
#         Classifylabel = [[] for _ in range(len(bestFeature))]
#         for sampleSet in dataclassifySet:
#             for sample in sampleSet:
#                 Classifylabel[dataclassifySet.index(sampleSet)].append(sample[-1])
#         print(Classifylabel)
#
#         # type the final label and delete the branch that has been labeled
#         for set in Classifylabel:
#             s = []
#
#             [s.append(i) for i in set if i not in s]
#             if len(s) == 1:
#                 print(bestFeature[Classifylabel.index(set)], 'has one unit label')
#                 print('s is', s)
#                 final_Label[bestFeature[Classifylabel.index(set)]] = s[-1]
#                 print('final label is', final_Label)
#                 del dataclassifySet[Classifylabel.index(set)]
#                 print('after delete, whole dataset = ', dataclassifySet)
#                 del bestFeature[Classifylabel.index(set)]
#                 print('after delete, best feature = ', bestFeature)
#                 del Classifylabel[Classifylabel.index(set)]
#                 print('after delete, dataset = ', Classifylabel)
#
#         # depth = 1  #
#         # input_depth = 3 #
#
#         # label based on the depth (when depth = input depth)
#         subLabel = {}
#         max_labelNum = 0
#         max_Label = []
#         if depth == input_depth:
#             for sub_set in Classifylabel:
#                 for subSetLabel in sub_set:
#                     if subSetLabel not in subLabel.keys():
#                         subLabel[subSetLabel] = 0
#                     subLabel[subSetLabel] += 1
#                 max_labelNum = 0
#                 max_Label = []
#                 for levelLabels in subLabel.keys():
#                     if subLabel[levelLabels] > max_labelNum:
#                         max_Label = levelLabels
#                         max_labelNum = subLabel[levelLabels]
#                 final_Label[bestFeature[Classifylabel.index(sub_set)]] = max_Label
#         print('1st level final_label is', final_Label)
#
#         # the second layer of tree
#         newDataClassSet = []
#         newBestFeat = []
#         newIndexBestFeat = []
#         for SecondTimeSub in Classifylabel:
#             print('Second=', dataclassifySet[Classifylabel.index(SecondTimeSub)])
#             newDataClassSet.append(dataclassifySet[Classifylabel.index(SecondTimeSub)])
#             returnfeatlength, returnbestFeature = BestFeature(dataclassifySet[Classifylabel.index(SecondTimeSub)])
#             print('new best feature is=', returnbestFeature)
#             newBestFeat.append(returnbestFeature)
#             newIndexBestFeat.append(returnfeatlength)
#         depth += 1
#         print('new depth is', depth)
#         print('the length is', len(newDataClassSet), 'the next layer dataset:', newDataClassSet)
#         print('the next layer best feature:', newBestFeat)
#         print('the next layer best feature length:', newIndexBestFeat)

####################################3##########################loop end##############

############################### first layer start here  ################
# the length of dataset and the dataset after split depend on the best feat
dataclassifySet = [[] for _ in range(len(bestFeature))]
#IndexBestFeat = 5
for BestSingleFeat in bestFeature:  # sample_data in train_data:
    for sample_data in train_data:
        if sample_data[IndexBestFeat] == BestSingleFeat:
            dataclassifySet[bestFeature.index(BestSingleFeat)].append(list(sample_data))
print('len(dataclassifySet)= ',len(dataclassifySet), 'dataclassifySet =', dataclassifySet, '\n')


# extract the the label from the dataset that is split based on the best feture
Classifylabel=[[] for _ in range(len(bestFeature))]
for sampleSet in dataclassifySet:
    for sample in sampleSet:
        Classifylabel[dataclassifySet.index(sampleSet)].append(sample[-1])
print(Classifylabel)

# type the final label and delete the branch that has been labeled
for set in Classifylabel:
    s = []

    [s.append(i) for i in set if i not in s]
    if len(s) == 1:
        print(bestFeature[Classifylabel.index(set)], 'has one unit label')
        print('s is', s)
        final_Label[bestFeature[Classifylabel.index(set)]] = s[-1]
        print('final label is', final_Label)
        del dataclassifySet[Classifylabel.index(set)]
        print('after delete, whole dataset = ', dataclassifySet)
        del bestFeature[Classifylabel.index(set)]
        print('after delete, best feature = ', bestFeature)
        del Classifylabel[Classifylabel.index(set)]
        print('after delete, dataset = ',  Classifylabel)


#depth = 1  #
# input_depth = 3 #

# label based on the depth (when depth = input depth)
subLabel={}
max_labelNum = 0
max_Label = []
if depth == input_depth:
    for sub_set in Classifylabel:
        for subSetLabel in sub_set:
            if subSetLabel not in subLabel.keys():
                subLabel[subSetLabel] = 0
            subLabel[subSetLabel] += 1
        max_labelNum = 0
        max_Label = []
        for levelLabels in subLabel.keys():
            if subLabel[levelLabels] > max_labelNum:
                max_Label = levelLabels
                max_labelNum = subLabel[levelLabels]
        final_Label[bestFeature[Classifylabel.index(sub_set)]] = max_Label
print('1st level final_label is', final_Label)


# the second layer of tree
newDataClassSet=[]
newBestFeat=[]
newIndexBestFeat=[]
for SecondTimeSub in Classifylabel:
    print('Second=', dataclassifySet[Classifylabel.index(SecondTimeSub)])
    newDataClassSet.append(dataclassifySet[Classifylabel.index(SecondTimeSub)])
    returnfeatlength, returnbestFeature = BestFeature(dataclassifySet[Classifylabel.index(SecondTimeSub)])
    print('new best feature is=', returnbestFeature)
    newBestFeat.append(returnbestFeature)
    newIndexBestFeat.append(returnfeatlength)
depth += 1
print('new depth is', depth)
print('the length is', len(newDataClassSet), 'the next layer dataset:', newDataClassSet)
print('the next layer best feature:', newBestFeat)
print('the next layer best feature length:', newIndexBestFeat)

############################ first layer start here ################


# train_data, attributes = createDataset()
#
# depth = 1  #
# input_depth = 3 #
# bestfeature = BestFeature(train_data)
# final_Label = {}

####################################loop start  here###################



# train_data, attributes = createDataset()
#
# depth = 1  #
# input_depth = 4#
#
# IndexBestFeat, bestFeature = BestFeature(train_data)
# print('bestFeature:', bestFeature)
# print(IndexBestFeat)
# unitIndexBestF = []
# unitIndexBestF.append(IndexBestFeat)
# print('unitIndexBestF=', unitIndexBestF)
# final_Label = {}
# unitTrainData=[]
# unitTrainData.append(train_data)
# print('unitTrainData=', unitTrainData)
# unitBestFeat=[]
# unitBestFeat.append(bestFeature)
# print('unitBestFeat=', unitBestFeat)
# buildTree(depth, input_depth, unitIndexBestF, unitBestFeat, unitTrainData, attributes, final_Label)














