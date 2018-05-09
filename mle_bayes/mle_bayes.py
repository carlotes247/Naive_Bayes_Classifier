#=======================
#IMPORT DECLARATIONS
#=======================
import math

#=======================
#DEFINITION OF VARIABLES 
#=======================

# User Input string
userInput = input("")
# Arities (num of possible values) for each variable
aritiesValuesTrainingSet = []
aritiesValuesTestSet = []
# The values of each class
classValuesTrainingSet = []
classValuesTestSet = []
# The probabilites of each class
classProbabilitties = []
# Path to load the training dataset
trainingDataPath = ""
testDataPath = ""
# Flag for debug code
debugFlag = False


#=======================
#DEFINITION OF FUNCTIONS 
#=======================

# Separation input string from user
def separateInputUser(inputString):
    # split entries by whitespaces
    separatedString = inputString.split()
    # return separated entries
    return separatedString

# Loads data from a path and prepares dataset in vectors passed in
def loadDataFromPath(path, aritiesVector, classValuesVector):
    # load data from text files
    with open(path) as traindata:
        data = traindata.readlines()

        # Convert data into class vectors
        for line in range(len(data)):
            # If it is the second line...
            if line == 1:
                # It is the line of arities
                aritiesVector = separateDataByVectors(data[line])    
                if debugFlag:
                    print ("Arities Values are: " + str(aritiesVector))
            # if it is further than the second line...
            if line > 1:
                # it is the class values
                auxClassValues = separateDataByVectors(data[line])
                # we add rows of vectors into the classValues vector (C,x1,x2,x3)
                classValuesVector.append(auxClassValues)
                if debugFlag:
                    print ("Values per class are: " + str(auxClassValues))

        # return both the aritiesVector and the classValuesVector
        return aritiesVector, classValuesVector

# Separate training data into integer vectors (assumming is formatted as a string)
def separateDataByVectors(dataset):
    vector = dataset
    # declare the integer vector to return
    vectorInt = []
    if debugFlag:   
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

    #return integer vector
    return vectorInt

# Separate training data between classes (assuming is formatted as integer vectors)
def separateDataByClass(dataset):
    # declare dictionary for classes and values
    classesDataVector = {}
    # We order the data in classes following the arity of the classes
    for i in range(aritiesValuesTrainingSet[0]):
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
    if debugFlag:    
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
    # we assume we have the dataset of a class already
    # create the summaries of the attributes per class [mean, stdDeviation]
    summariesAttributes = [(mean(attribute), stdDeviation(attribute)) for attribute in zip(*dataset)]
    # remove the value for the class (making vector one position shorter) (position 0 is class value)
    del summariesAttributes[0]
    # return summary with mean and std Deviation
    return summariesAttributes      

#Summarises all attribute values (mean and std Deviation) per class (WE ASSUME THE DATASET IS ALREADY SEPARATED BETWEEN CLASSES)
def summariseValuesByClass(dataset):
    if debugFlag:
        print("Summarising the following dataset: " + str(dataset))
    # Create the dictionary
    summariesClasses = {}
    # Fill dictionary per classValues iterating over the list of separatedClasses
    for classValue, instances in dataset.items():
        # Get the summary (mean, std Deviation) for each attribute in the list
        summariesClasses[classValue] = summariseAttributeValues(instances)
    # Return the filled list of means and std deviations per classes
    return summariesClasses

# Calculate the probability that a value belongs to a class using a gaussian function given an attribute value, mean and std deviation
def calculateProbability(x, mean, stdDeviation):
    if stdDeviation == 0:
        stdDeviation = 0.00001
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
            # get mean and std deviation from the summary for each class entry
            mean, stdDeviation = classSummaries[i]
            # Declare x 
            x = vector[i]
            # if vector[i] is still a vector...
            if type(x) is list:
                # we make sure to be checking all the attribute values inside that vector
                for j in range(len(x)):
                    # access the attribute value for that class
                    y = x[j]
                    probabilities[classValue] *= calculateProbability(y, mean, stdDeviation)
            # If it is not a list calculate the class probability given all the data
            else:
                probabilities[classValue] *= calculateProbability(x, mean, stdDeviation)
    # return dictionary of class probabilities
    return probabilities

# Calculate class prediction based on the largest probability of a data instance belonging to a class
def predictClass(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    # select the max of the probabilities
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

# Function to test the model following task specifications
def testModel(testSet):
    # loop through the test Set and output probabilities per entry
    for i in range(len(testSet)):
        attribute = testSet[i]
        classValueAttribute = attribute [0]
        summaryAttribute = summary.get(classValueAttribute, i)
        # calculate class probability
        resultClassProbability = calculateClassProbabilities(summary, attribute)
        # predict class
        resultClassPredicted = predictClass(summary, attribute)
        # output result as specified in task submission
        print ("P(C="+str(classValueAttribute)+" | X1="+str(attribute[1])+" | X2="+str(attribute[2])+" | X3="+str(attribute[3])+") = "+str(resultClassProbability)+" | Class Prediction="+str(resultClassPredicted))
    return True


#=======================
#MAIN PART OF THE CODE
#=======================

# formatting user input into strings
pathsUserInput = separateInputUser(userInput)
trainingDataPath = pathsUserInput[0]
testDataPath = pathsUserInput[1]

# load training data set
print("Loading training and test data", end="")
aritiesValuesTrainingSet, classValuesTrainingSet = loadDataFromPath(trainingDataPath, aritiesValuesTrainingSet, classValuesTrainingSet)
# load test data set
aritiesValuesTestSet, classValuesTestSet = loadDataFromPath(testDataPath, aritiesValuesTestSet, aritiesValuesTestSet)
print("")
print("=================")
print ("Arities training values are: " + str(aritiesValuesTrainingSet))
print ("Class training values are: " + str(classValuesTrainingSet))
print("Training classifier...")
classValuesTrainingSet = separateDataByClass(classValuesTrainingSet)
print("=================")
print ("Arities test values are: " + str(aritiesValuesTestSet))
print ("Class test values are: " + str(classValuesTestSet))
print("=================")

# prepare model
summary = summariseValuesByClass(classValuesTrainingSet)

# test model
testModel(classValuesTestSet)

print("=================")
# calculate accuracy classifier
predictions = getPredictionsClasses(summary, classValuesTestSet)
accuracy = getAccuracyClassification(classValuesTestSet, predictions)
print('Accuracy: ' + str(accuracy))
