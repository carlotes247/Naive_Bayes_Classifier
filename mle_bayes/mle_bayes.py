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
print("Summary of " + str(numbers) + " : mean=" + str(mean(numbers)) + " , stdDev=" + str(stdDeviation(numbers)))
