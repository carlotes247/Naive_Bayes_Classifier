#=======================
#IMPORT DECLARATIONS
#=======================
import math

#=======================
#DEFINITION OF VARIABLES 
#=======================

# Arities (num of possible values) for each variable
aritiesValues = []
# The values of each class
classValues = []
# The probabilites of each class
classProbabilitties = []
# Path to load the training dataset
trainingDataPath = "traindata1.txt"

#=======================
#DEFINITION OF FUNCTIONS 
#=======================

# Separate training data into integer vectors (assumming is formatted as a string)
def separateDataByVectors(dataset):
    vector = dataset
    # declare the integer vector to return
    vectorInt = []
    # show status update
    print ("Formatting data", end="")
    # for each of the arities...
    for entry in range(len(vector)):
        # try to convert the value into an integer
        try:
            number = int(vector[entry])
            vectorInt.append(number)                   
        # if error... 
        except:
            # Print status update (it was not an integer value in the string)      
            print (".", end="")

    # break line in console
    print("")
    #return integer vector
    return vectorInt

# Separate training data between classes (assuming is formatted as integer vectors)
def separateDataByClass(dataset):
    # declare dictionary for classes and values
    classesDataVector = {}
    # We order the data in classes following the arity of the classes
    for i in range(aritiesValues[0]):
        # we go through the entire data in search for class arity i
        for j in range(len(dataset)):
            # Get the first vector in the list
            vector = dataset[j]
            # search for the class value i (first value is the class value)
            if (vector[0] == i):
                # create entry in dictionary according to i value
                if i not in classesDataVector:
                    classesDataVector[i] = []             
                # we add this vector now to the list because it matches the class value we search for!
                classesDataVector[i].append(vector)
    # Debug the new order of classes
    print("New order of classValues: " + str(classesDataVector))
    return classesDataVector

# Simple mean calculation given a set of values    
def mean (values):
    return sum(values)/float(len(values))

# Standard Deviation calculation
def stdDeviation (values):
    # We first calculate the mean
    average = mean(values)
    # We then use the mean to calculate the variance (using the N-1 bias correction because the mean is unknown)
    variance = sum([pow(x-average,2) for x in values])/float(len(values)-1)
    # return the square root of the variance (standard deviation)
    return math.sqrt(variance)

# Calculate mean and std Deviation for each attribute in a given list of class values
def summariseAttributeValues(dataset):
    # create the summaries of the attributes per class [mean, stdDeviation]
    summariesAttributes = [(mean(attribute), stdDeviation(attribute)) for attribute in zip(*dataset)]
    # remove the value for the class (making vector one position shorter) (position 0 is class value)
    del summariesAttributes[0]
    # return summary with mean and std Deviation
    return summariesAttributes      

#Summarises all attribute values (mean and std Deviation) per class
def summariseValuesByClass(dataset):
    # We separate the data by classes
    separatedClasses = separateDataByClass(dataset)
    # Create the dictionary
    summariesClasses = {}
    # Fill dictionary per classValues iterating over the list of separatedClasses
    for classValue, instances in separatedClasses.items():
        # Get the summary (mean, std Deviation) for each attribute in the list
        summariesClasses[classValue] = summariseAttributeValues(instances)
    # Return the filled list of means and std deviations per classes
    return summariesClasses

# Calculate the probability that a value belongs to a class using a gaussian function given an attribute value, mean and std deviation
def calculateProbability(x, mean, stdDeviation):
    # calculate the exponent 
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdDeviation,2))))
    # return the probability
    return (1 / (math.sqrt(2*math.pi) * stdDeviation)) * exponent

def calculateClassProbabilities(summariesAttributes, vector):
    # define the probabilities dictionary
    probabilities = {}
    # Go through the list of means and std deviations
    for classValue, classSummaries in summariesAttributes.items():
        # declare the entry for the classValue as an integer
        probabilities[classValue] = 1
        # go through the class summaries (mean and std deviation)
        for i in range(len(classSummaries)):
            # get mean and std deviation for each class
            mean, stdDeviation = classSummaries[i]
            # Declare x 
            x = vector[i]
            # calculate the class probability given all the data
            probabilities[classValue] *= calculateProbability(x, mean, stdDeviation)
    # return dictionary of class probabilities
    return probabilities

# Calculate class prediction based on the largest probability of a data instance belonging to a class
def predictClass(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

# Predict classes given a test dataset and a trained summary of means and std deviations
def getPredictionsClasses(summaries, testSet):
    # declare predictions list
    predictions = []
    # go through the test dataset
    for i in range(len(testSet)):
        # predict to which class is the entry in the set belonging to
        result = predictClass(summaries, testSet[i])
        # add result class to list
        predictions.append(result)
    # return list of class predictions
    return predictions
 
# Calculate the accuracy of predictions   
def getAccuracyClassification(testSet, predictions):
    # declare correct predictions counter
    correctPredictions = 0
    # go through the test dataset
    for x in range(len(testSet)):
        # if the class value of the testSet is equal to the predicted one...
        if testSet[x][0] == predictions[x]:
            # increase counter by one
            correctPredictions += 1
    # return accuracy as a percentage
    return (correctPredictions/float(len(testSet))) * 100.0

#=======================
#LOADING DATA FROM FILES 
#=======================

# load data from text files
with open(trainingDataPath) as traindata:
    data = traindata.readlines()

    # Convert data into class vectors
    for line in range(len(data)):
        # If it is the second line...
        if line == 1:
            # It is the line of arities
            aritiesValues = separateDataByVectors(data[line])    
            print ("Arities Values are: " + str(aritiesValues))
        # if it is further than the second line...
        if line > 1:
            # it is the class values
            auxClassValues = separateDataByVectors(data[line])
            # we add rows of vectors into the classValues vector (C,x1,x2,x3)
            classValues.append(auxClassValues)
            print ("Values per class are: " + str(auxClassValues))

#=======================
#MAIN PART OF THE CODE
#=======================

print ("Arities values are: " + str(aritiesValues))
print ("Class values are: " + str(classValues))
classValues = separateDataByClass(classValues)
numbers = [1,2,3,4,5]
#print("Summary of " + str(numbers) + " : mean=" + str(mean(numbers)) + " , stdDev=" + str(stdDeviation(numbers)))

# code to test attribute summaries
#dataset = [[0,1,20], [1,2,21], [0,3,22]]
#summary = summariseAttributeValues(dataset)
#print('Attribute summaries: ' + str(summary))

# code to test class summaries
#dataset = [[1,1,20], [0,2,21], [1,3,22], [0,4,22]]
#summary = summariseValuesByClass(dataset)
#print('Summary by class value: ' + str(summary))

# code to test probability of belonging to a class
#x = 71.5
#mean = 73
#stdev = 6.2
#probability = calculateProbability(x, mean, stdev)
#print('Probability of belonging to this class: ' + str(probability))

# code to test the class probabilities
#summaries = {0:[(1, 0.5)], 1:[(20, 5.0)]}
#inputVector = [1.1, '?']
#probabilities = calculateClassProbabilities(summaries, inputVector)
#print('Probabilities for each class: ' + str(probabilities))

# code to test the predictions
#summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
#inputVector = [1.1, '?']
#result = predictClass(summaries, inputVector)
#print('Prediction: ' + str(result))

# code to test the getPredictions of entire dataset
#summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
#testSet = [[1.1, '?'], [19.1, '?']]
#predictions = getPredictionsClasses(summaries, testSet)
#print('Predictions: ' + str(predictions))

# code to test the accuracy of predictions
testSet = [['a',1,1,1], ['a',2,2,2], ['b',3,3,3]]
predictions = ['a', 'a', 'a']
accuracy = getAccuracyClassification(testSet, predictions)
print('Accuracy: ' + str(accuracy))