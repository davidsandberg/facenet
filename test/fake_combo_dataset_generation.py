import json
import numpy as np

output_path = ('/home/iolie/thorn/sharon/MINIVECS')
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


### I just want to randomly select two people and combine some of their photos
## really wuld want to make about 10,000 of these
# def make_two_person_set():
fake_combos_scores = []

fake_women_combo_counter = 0
while fake_women_combo_counter < 10000:
    holder = np.asarray(d.keys())
    random_numbers = np.random.randint(len(holder), size=2)  ## how to ensure it's not the same number?

    if random_numbers[0] != random_numbers[1]:
        selector_a = holder[random_numbers[0]]
        a = d[selector_a].keys()

        woman_1_sample = np.empty(shape=(2, 128))
        if len(a) > 1:
            #print("len a" + "  " + str(len(a)))
            woman_1 = []
            for x in a:
                woman_1.append(d[selector_a][x])
            all_woman1_array = np.asarray(woman_1)
            #print("all_woman1_array" , all_woman1_array.shape)
            woman_1_sample = all_woman1_array[0:2, :]
            #print(woman_1_sample.shape)

        selector_b = holder[random_numbers[1]]

        b = d[selector_b].keys()
        woman_2_sample = np.empty(shape=(2, 128))
        if len(b) > 1:
            #print("len b" +  "  " + str(len(b)))
            woman_2 = []
            for y in b:
                woman_2.append(d[selector_b][y])
            all_woman2_array = np.asarray(woman_2)
            woman_2_sample = all_woman2_array[0:2, :]
            #print(woman_2_sample.shape)

        fake_combo_woman = np.append(woman_1_sample, woman_2_sample, axis = 0)
        #print(fake_combo_woman.shape)

        if len(fake_combo_woman) == 4:
            sim = get_sim(fake_combo_woman)
            #print(blep)
            simapp = np.append(selector_b, sim)
            names_and_score = np.append(selector_a, simapp)
            #print(blepo)
            fake_combos_scores.append(name_and_score)
            fake_women_combo_counter = fake_women_combo_counter +1
#return fake_combos_scores

#make_two_person_set()
print(len(fake_combos_scores))
aa = np.asarray(fake_combos_scores)
print(aa.shape)

