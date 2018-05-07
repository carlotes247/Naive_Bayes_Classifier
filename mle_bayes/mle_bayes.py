#=======================
#DEFINITION OF VARIABLES 
#=======================

# Arities (num of possible values) for each variable
aritiesValues = []
# The values of each class
classValues = []
# The probabilites of each class
classProbabilitties = []

# Separate training data into classes (assumming is formatted as a string)
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

#=======================
#LOADING DATA FROM FILES 
#=======================

# load data from text files
with open("traindata1.txt") as traindata:
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
print ("Class values are: " + str(classValues[6]))


