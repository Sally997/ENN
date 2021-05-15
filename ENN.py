import numpy as np
from collections import Counter
def ENN(dataset, labels, k):
    totalset = []
    indsOftotalset = []
    #list of lists, each mini list contains the distances between each point and other data point [[0.4,0.04,0.3,0.7],[0.5,0.4,...],...]
    distances = []
    #calculating the distances for each data point
    for ind in range(0, len(dataset)):
        #the list containing the distances of the individual data point ind
        newDist = []
        #using the already calculated distances, for instance the point number 3 of the data its distances from points 0,1,2 are already calculated in mini lists 0,1,2[existed] in the index 2[ind-1]
        for existed in range(0, ind):
            newDist.append(distances[existed][ind-1])
        #continuing to calculate the remaining distances
        for nxtind in range(ind+1, len(dataset)):
            dist = np.linalg.norm(dataset[ind] - dataset[nxtind])
            newDist.append(dist)
        distances.append(newDist)

        removedInstances = []
        Y=[]
        #get the k minimum distances(k nearest neighbors)
        knn=sorted(newDist)[:k]
        knnind=[]
        #get the indeces of the k nearest neighbors
        for x in knn:
            #if [0.5,0.8,0.6,0.4,...] is the list of distances for data point 3(ind=2)
            #the first two values (0.5,0.8) represent respectively the distances between 2 and 0, 2 and 1
            if x<ind:
                knnind.append(newDist.index(x))
            else:
                # but the value 0.6 represent the distance between 2 and 3 that's why we need to add one to the index to get the right data point index
                knnind.append(newDist.index(x)+1)
        #get the labels(classes) of the selected data points as nearest neighbors
        print('knnin', knnind)
        for lab in knnind:

            Y.append(labels[lab])
        print("Y as it's the list of labels for the k nearest neighbor", Y)
        #if Y is not an empty list
        if Y:
            c = Counter(Y)
            #print('counter values', c, c.most_common())
            value, count = c.most_common()[0]
            print('the class of the data point ind is', labels[ind],'the common class among the k nearest neighbors is', value)
            #if the class of the data point ind is not the same as the common class among the k nearest neighbor of it then it's noise
            if labels[ind] != value:
                totalset.append(dataset[ind])
                indsOftotalset.append(ind)
                print('this point is noise')
            else:
                print('this point is not noise')
    #remove the data points that are noise using their index
    newDataset = np.delete(dataset, indsOftotalset)
    return newDataset, totalset

from sklearn.datasets import make_classification
X, Y = make_classification(n_samples=10000, n_features=20, n_redundant=0, n_classes=4, n_clusters_per_class=1, flip_y=0, random_state=1)
#print(X, Y)
newDataset, noise=ENN(X, Y, 3)
print('num of noise points is', len(noise))

print(newDataset)