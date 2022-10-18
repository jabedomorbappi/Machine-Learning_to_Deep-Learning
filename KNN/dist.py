import numpy as np 

def euclian_distance(p1,p2):
    dist=0
    for i in range(len(p1)):
        dist+=(p1[i]-p2[1])**2
    euclidian_dist=np.sqrt(dist)
    return euclidian_dist