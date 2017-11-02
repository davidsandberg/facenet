import json
import numpy as np

output_path = ('/home/iolie/thorn/sharon/recheckvecs')
d = json.load(open(output_path))

def compare(x,y):
    dist = np.sqrt(np.sum(np.square(np.subtract(x, y))))
    return (dist)

from itertools import combinations
def get_sim(vecs):
    n=len(vecs)
    if(n<=1):
        return 1,0,0,n
    combe12=list(combinations(range(n),2))
    ncomb=len(combe12)
    sim=np.zeros(ncomb)
    for i in range(ncomb):
        e12=combe12[i]
        sim[i]=compare(vecs[e12[0]],vecs[e12[1]].T)
    mean=np.mean(sim)
    std=np.std(sim)
    min_sim=np.min(sim)
    return mean,min_sim,std,n

def print_summary_stats(scores):
    print("Average distance: " + str(np.average(scores)))
    print("Minimum distance: " + str(np.min(scores)))
    print("Maximum distance: " + str(np.max(scores)))
    return

## in dictionary, for each subkey, give me a numpy array of the subkey's corresponding vector list
## ALSO get all possible combinations of those vectors
## get the stats for those, along with some bonuses

people_scores = []
for person in d:
    a = d[person].keys()
    if len(a) > 1:
        #print(a)
        temp = []
        for x in a:
            temp.append(d[person][x])
        temparray = np.asarray(temp)
        #print(temparray.shape)
        bep = get_sim(temparray)
        bepo = np.append(person, bep)
        print(bepo)
        people_scores.append(bepo)


print("hello")
pscores = np.asarray(people_scores)
x = pscores[:, 1].astype(np.float32)
print(x)
print_summary_stats(x)

print(people_scores[np.argmin(x)][0])
print(people_scores[np.argmax(x)][0])

#holder = np.asarray(d.keys())
#print(holder)
#random_numbers = np.random.randint(len(holder), size=2)
#selector_a = holder[random_numbers[0]]
#print(selector_a)
#a = d[selector_a].keys()
#print(a)
#testarray = np.asarray(test)
#print(testarray)
#print(testarray.shape)



