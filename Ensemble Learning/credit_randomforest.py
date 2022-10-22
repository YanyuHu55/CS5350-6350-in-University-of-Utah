import random
from math import log
import numpy as np
import math
import statistics
import matplotlib.pyplot as plt

import operator
import copy

# create the training data set
def createDataset():
    dataSet = np.loadtxt('train.csv', dtype=str, delimiter=",")
    attributes = ['X1', 'X2', 'X3', 'X4','X5','X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23']
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
    # print('bestfeatures dataset lengtth', len(dataSet),'best feature is', dataSet, )
    numFeature = len(dataSet[0]) -1# the number of attributes
    # print('number of feature', numFeature)
    baseEntropy=shannonEnt(dataSet) # the Shannon Entropy of the training dataset
    # print(baseEntropy)
    bestInfoGain = 0.0  # Initial condition of Information gain = 0
    bestFeature = -1  # the index of best feature in the attribute set
    # print(numFeature)
    gain = []
    Attribute = []
    for i in range(numFeature):
        featList = [example[i] for example in dataSet]# get the ith attribute of all dataset
        # print('The', i, 'th ', 'feat list is', featList)
        # print('featlist len=', len(featList), 'featlist type', type(featList))
        # print('featvalue=', featValue)
        # featValue = set(np.array(featList))# create a set
        featValue = []
        [featValue.append(i) for i in featList if i not in featValue]
        # print('featValue is', featValue)
        attributes = list(featValue)
        # print('attribute is', attributes)
        # print(featList)
        newEntropy = 0.0  # Initial condition of Entropy = 0
        for value in featValue:  # be focus
            subDataSet = splitDataSet(dataSet, i, value) # call the function to partition the dataset
            # print('The subDataSet is ', subDataSet)
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

    # print('The best attribute is: %s' % Attribute[gain.index(max(gain))])
    # print('maximum information gain is ', gain.index(max(gain)))

    return gain.index(max(gain)), Attribute[gain.index(max(gain))], Attribute



# the length of dataset and the dataset after split depend on the best feat
def SplittheData(train_data, bestFeature, IndexBestFeat):
    dataclassifySet = [[] for _ in range(len(bestFeature))]
    # IndexBestFeat = 5
    for BestSingleFeat in bestFeature: # sample_data in train_data:
        # label_KeyIndex[IndexBestFeat] = BestSingleFeat
        # final_Label[tuple(label_KeyIndex)] = 0
        for sample_data in train_data:
            if sample_data[IndexBestFeat] == BestSingleFeat:
                dataclassifySet[bestFeature.index(BestSingleFeat)].append(list(sample_data))
    # print('len(dataclassifySet)= ', len(dataclassifySet), 'dataclassifySet =', dataclassifySet, '\n')
    # print('Label is', final_Label)
    return dataclassifySet


# extract the the label from the dataset that is split based on the best feture
def ExtractLabel(bestFeature, dataclassifySet, final_Label, IndexBestFeat, label_KeyIndex):
    Classifylabel = [[] for _ in range(len(bestFeature))]
    for sampleSet in dataclassifySet:

        for sample in sampleSet:
            Classifylabel[dataclassifySet.index(sampleSet)].append(sample[-1])
    # print('ClassifyLabel is ', Classifylabel)

    # type the final label and delete the branch that has been labeled
    for set in Classifylabel:
        s = []

        [s.append(i) for i in set if i not in s]
        if len(s) == 1:
            # print(bestFeature[Classifylabel.index(set)], 'has one unit label')
            # print('s is', s)
            label_KeyIndex[IndexBestFeat] = bestFeature[Classifylabel.index(set)]
            final_Label[tuple(label_KeyIndex)] = s[-1]
            # print('if the attributes is', bestFeature[Classifylabel.index(set)],'. final label of this branch is', final_Label)
            del dataclassifySet[Classifylabel.index(set)]
            # print('after delete, whole dataset = ', dataclassifySet)
            del bestFeature[Classifylabel.index(set)]
            # print('after delete, best feature = ', bestFeature)
            del Classifylabel[Classifylabel.index(set)]
            # print('length of ClassifyLabel', len(Classifylabel), 'after delete, dataset = ', Classifylabel)
    return final_Label, dataclassifySet, bestFeature, Classifylabel



# label based on the depth (when depth = input depth)
def DepthDetect(Classifylabel, bestFeature,label_KeyIndex, new_IndexBestFeat):
    subLabel = {}
    max_labelNum = 0
    max_Label = []
    # if depth == input_depth:
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
        label_KeyIndex[new_IndexBestFeat] = bestFeature[Classifylabel.index(sub_set)]
        final_Label[tuple(label_KeyIndex)] = max_Label
    # print('1st level final_label is', final_Label)
    return final_Label

#####################################  first layer   ################################



####################################3##########################loop end##############

############################### first layer start here  ################
def tree(input_depth, label_KeyIndex,dataclassifySet,final_Label):
    all_labelkey = []
    for i in range(0, input_depth):
        Data_Set = []
        a = -1

        for sublayer in dataclassifySet:
            # print('all_keylabel is ', all_labelkey)
            if i > 0:
                label_KeyIndex = all_labelkey[a]
            new_IndexBestFeat, new_BestFeature, artribute = BestFeature(sublayer)
            new_dataClassifySet = SplittheData(sublayer, new_BestFeature, new_IndexBestFeat)
            final_Label, new_dataClassifySet, new_BestFeature, new_ClassifyLabel = ExtractLabel(new_BestFeature,
                                                                                                new_dataClassifySet,
                                                                                                final_Label,
                                                                                                new_IndexBestFeat,
                                                                                                label_KeyIndex)
            # print(i, 'th label_key = ', label_KeyIndex)
            if i == input_depth - 1:
                final_Label = DepthDetect(new_ClassifyLabel, new_BestFeature, label_KeyIndex, new_IndexBestFeat)
            Data_Set = Data_Set + new_dataClassifySet
            # print('label_key = ', label_KeyIndex)
            # print ('new best feature is ', new_BestFeature)
            for sub_bestFeature in new_BestFeature:
                # print('all_labelkey.append(label_KeyIndex)', all_labelkey)
                # print('subbest feature is:', sub_bestFeature)
                label_KeyIndex[new_IndexBestFeat] = sub_bestFeature
                # print('all_labelkey.append(label_KeyIndex)', all_labelkey)
                # print('label_KeyIndex[new_IndexBestFeat]',label_KeyIndex[new_IndexBestFeat])
                all_labelkey.append(copy.deepcopy(label_KeyIndex))
                # print('all_labelkey.append(label_KeyIndex)', all_labelkey)
            # print ('all_keylabel is ', all_labelkey)
            a += 1

        dataclassifySet = Data_Set
        i += 1
        # print(i, 'th final label : ',final_Label)
    return final_Label

# input test data
def test_data():
    test_attribut = []
    test_label = []
    with open('test.csv', 'r') as file:
        for line in file:
            s = {}
            term = line.strip().split(',')
            sub_testAttr = term[:-1]
            sub_testLab = term[-1]
            test_attribut.append(copy.deepcopy(sub_testAttr))
            test_label.append(copy.deepcopy(sub_testLab))
    # print('test_attribute is',test_attribut)
    # print('label is', test_label)
    return test_attribut, test_label

def traindata():
    train_attribut = []
    train_label = []
    with open('train.csv', 'r') as file:
        for line in file:
            s = {}
            term = line.strip().split(',')
            sub_testAttr = term[:-1]
            sub_testLab = term[-1]
            train_attribut.append(copy.deepcopy(sub_testAttr))
            train_label.append(copy.deepcopy(sub_testLab))
    # print('test_attribute is',test_attribut)
    # print('label is', test_label)
    return train_attribut, train_label

# bagging label
def randomforest(train_data, input_depth):
    for t in range(0, 500):
        label_KeyIndex = ['none' for _ in range(len(attributes))]
        final_Label = {}
        label_set = []
        draw_sample = [random.choices(train_data, k=1000)]
        # print('draw data: ', draw_sample)
        final_Label = tree(input_depth, label_KeyIndex, draw_sample, final_Label)
        label_set.append(copy.deepcopy(final_Label))
    print('final label:', label_set)
    return label_set

def value_final(label_set):
    final_Keyset = []
    final_Valueset = []
    for final_Label in label_set:
        finalLabel_key = []
        finalLabel_value = []
        for i in final_Label.keys():
            print('finalLabel_key', finalLabel_key)
            finalLabel_key.append(copy.deepcopy(i))
            # print('final label key: ', finalLabel_key)
            finalLabel_value.append(copy.deepcopy(final_Label[i]))
            # print('final label value: ', finalLabel_value)
        final_Keyset.append(copy.deepcopy(finalLabel_key))
        final_Valueset.append(copy.deepcopy(finalLabel_value))
    return final_Keyset, final_Valueset

# extract test label and train label
def extract_label(test_attribut, final_Keyset, test_label, final_Valueset):
    compare_Labelset=[]
    compare_testset=[]
    compare_trainset=[]
    for i in range(0,1):
        finalLabel_key = final_Keyset[i]
        finalLabel_value = final_Valueset[i]
        compare_Label = []
        labelPair = []
        b = 0
        for every in test_attribut:
            # print('every', every)
            for subKey in finalLabel_key:
                # print('subkey', subKey)
                i = 0
                for a in subKey:
                    # print('a', a)

                    if a == 'none':
                        i += 1
                        continue
                    elif a == every[subKey.index(a)]:
                        i += 1
                        continue
                    else:
                        b = 1
                        break
                # print('i = ', i)
                if i == len(every):
                    labelPair = []
                    labelPair.append(copy.deepcopy(test_label[test_attribut.index(every)]))
                    labelPair.append(copy.deepcopy(finalLabel_value[finalLabel_key.index(subKey)]))
                    # print('Label Pair is ', labelPair)
                    compare_test.append(copy.deepcopy(test_label[test_attribut.index(every)]))
                    compare_train.append(copy.deepcopy(finalLabel_value[finalLabel_key.index(subKey)]))

                    # print('compare label1 is ', compare_Label)
                    compare_Label.append(copy.deepcopy(labelPair))
                    b = 0
                    break
            if b == 1:
                fakeLabel = []
                fakeLabel.append(copy.deepcopy(test_label[test_attribut.index(every)]))
                fakeLabel.append('no')
                compare_Label.append(copy.deepcopy(labelPair))
                compare_test.append(copy.deepcopy(test_label[test_attribut.index(every)]))
                compare_train.append('no')
                # print('compare label2 is ', compare_Label)
                b = 0
        compare_Labelset.append(copy.deepcopy(compare_Label))
        compare_testset.append(copy.deepcopy(compare_test))
        compare_trainset.append(copy.deepcopy(compare_train))

    return compare_Labelset, compare_testset, compare_trainset

train_data, attributes = createDataset()
train_data = np.array(train_data).tolist()
dataclassifySet = []
dataclassifySet.append(train_data)
print('train label', dataclassifySet)


depth = 0  #
input_depth = 10#

final_Label = {}
# final_Label=tree(input_depth, label_KeyIndex,dataclassifySet,final_Label)

test_attribut = []
test_label = []
compare_test = []
compare_train = []
no_test = 0
yes_test = 0

test_attribut, test_label = test_data()


label_set = randomforest(train_data, input_depth)


final_Keyset, final_Valueset = value_final(label_set)
print('final key set:',final_Keyset)
compare_Labelset, compare_testset, compare_trainset = extract_label(test_attribut, final_Keyset, test_label, final_Valueset)

