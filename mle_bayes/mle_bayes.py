#=======================
#DEFINITION OF VARIABLES 
#=======================

# Arities (num of possible values) for each variable
classArity = 0
x1Arity = 0
x2Arity = 0
x3Arity = 0
# The values of each class
classValues = {}
xValues = {}
# The probabilites of each class
classProbabilitties = {}



# Separate training data into classes
def separateDataByClass(dataset):
    # create empty vector for the separated entry
    separatedVector =[]  

    # for i in range(dataset.lenght)
    for i in range(len(dataset)):
        # get the vector data of each line in the dataset
        vectorData = dataset[i]
        # if the entry -1 is not in the separated entry...
        if (vectorData[-1] not in separatedVector):
            # get the last attribute which will be the class value
            separatedVector[vectorData[-1]] = []
        separatedVector [vectorData[-1]].append(vectorData)

    return separatedVector

#=======================
#LOADING DATA FROM FILES 
#=======================

# load data from text files
with open("traindata1.txt") as traindata:
    data = traindata.readlines()

    for line in range(len(data)):
        # If it is the second line...
        if line == 1:
            # It is the line of arities
            vector = data[line]
            print ("Arities are: " + data[line])
            # for each of the arities...
            for entry in range(len(vector)):
                # if it is the first arity...
                if entry == 0:
                    # it is the class arity!
                    print ("Class arity is " + vector[entry])
                # if it is not the first entry...
                else:
                    # it is each of the class variables arities!
                    print ("X" + str(entry) + " arity is " + vector[entry])

        if line > 1:
            print ("Values per class are: " + data[line])
        #print (data[line])
        #if line == 0 :
        #    entry = line.split()
        #    print (entry)
        



#=======================
#MAIN PART OF THE CODE
#=======================

#print (separateDataByClass(data))


