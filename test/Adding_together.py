import os
import json
import itertools
import numpy as np
import PIL.Image
from PIL import ImageFont
from PIL import ImageDraw
from sklearn.svm import SVC
from IPython.display import Image
from itertools import combinations
from IPython.display import display
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV as GSCV

## normalise the vector set before we start????

def compare(x,y):
    dist = np.sqrt(np.sum(np.square(np.subtract(x, y))))
    return (dist)

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

def make_real_person_data():
    single_person=[]
    for person in d:
        a = d[person].keys()
        if len(a) > 1:
            temp = []
            for x in a:
                temp.append(d[person][x])
            temparray = np.asarray(temp)
            # print(temparray.shape)
            singsim = get_sim(temparray)
            singsim_app = np.append(person, singsim)
            singsim_app2 = np.append(person, singsim_app)
            singsim_app_label = np.append("0", singsim_app2)
            #print(singsim_app_label)
            single_person.append(singsim_app_label)
    return single_person


def make_fake_combos(max_size):   # maybe start with 10k
    fake_combos_scores = []
    fake_women_combo_counter = 0
    while fake_women_combo_counter < max_size:
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
                simapp = np.append(selector_b, sim)
                name_and_score = np.append(selector_a, simapp)
                flag_name_score = np.append("1", name_and_score)
                #print(flag_name_score)
                fake_combos_scores.append(flag_name_score)
                fake_women_combo_counter = fake_women_combo_counter +1

    return fake_combos_scores


output_path = ('/home/iolie/thorn/sharon/recheckvecs')
d = json.load(open(output_path))

combo_people = make_fake_combos(10000)
print(len(combo_people))
combo_people_array = np.asarray(combo_people)
print(combo_people_array.shape)

single_person = make_real_person_data()
print(len(single_person))
bb = np.asarray(single_person)
clipped_single_person_array = bb[:10000, :]
print(clipped_single_person_array.shape)

real_and_fake = np.append(combo_people_array, clipped_single_person_array, axis=0)
print(real_and_fake.shape)


# split for train and test
data_train, data_test, key_train, key_test = train_test_split(real_and_fake[:, 3:6].astype(np.float32), real_and_fake[:, 0].astype(np.float32), test_size=0.01, random_state=42)
#print(data_train.shape)
#print(data_train[:2, ])
#print(data_test.shape)
#print(key_train.shape)
#print(key_train[0:3])
#print(key_test.shape)



#parameters = {'kernel':('linear', 'rbf'), 'C': np.logspace(-2, 1, 6)} ## gets to 96% with 10000 each
# other tries!
# parameters = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 1e-2, 1 ,1e2], 'C': np.logspace(-2, 1, 4)},  {'kernel': ['linear'], 'C': np.logspace(-2, 1, 4)}]
svm_model = SVC(probability = True) # this model trains to 95% accuracy with 10000 or each samples, and 96% accuracy with 20000 of each
# parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},  {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]} gets to 93%

#svm_model = GSCV(SVC(probability=True), parameters)

svm_model.fit(data_train, key_train)
print(svm_model.score(data_test, key_test))

test = svm_model.predict_proba(bb[15000:16000, 3:6])
print(test[:50, :])
#print((bb[15000:16000, 1]))
test_with_names = np.append(bb[15000:16000, 0:2], test, axis=1)
#test_with_names = np.vstack((bb[15000:16000, 1], test))
print(test_with_names.shape)
print(test_with_names[:50, :])

image_path = ('/home/iolie/thorn/sharon/bp_aug_2017_dataset_crop_to_folder')

def tile_faces(key, odds):
    pics_to_display = []
    for x in os.listdir(os.path.join(image_path, key)):
        a = os.path.join(image_path, key, x)
        b = PIL.Image.open(a)
        pics_to_display.append(b)

    width = pics_to_display[0].width
    total_width = width * (len(pics_to_display)+2)
    height = pics_to_display[0].height

    hm = PIL.Image.new('RGB', (width*2, height))
    hmm = ImageDraw.Draw(hm)
    hmm.text((20, 70), (str(key + "     " + odds)))
    pics_to_display.append(hm)

    new_im = PIL.Image.new('RGB', (total_width, height))
    x_offset = 0

    print("Building image with width " + str(width) + ", height " + str(height) + ", from " + str(len(pics_to_display)) + " images, based on a frame of shape ")
    for im in pics_to_display:
        new_im.paste(im, (x_offset, 0))
        x_offset += width

    print ("Width = " + str(width) + ", height = " + str(height))
    return(new_im)


mega_im = PIL.Image.new('RGB', (182*10, 182*30))
y_offset = 0
for x in range(0,30):
    woo = tile_faces(test_with_names[x,1], test_with_names[x,2])
    mega_im.paste(woo, (y_offset, 0))
    y_offset += 182


#aa = tile_faces("0A0A937C-016C-49E6-A9CA-480292B491BC").show()
# really would want to add them all to one page with the odds of them being the same person listed alongside


#def show_image_set(key, height=160):
#    for x in os.listdir(os.path.join(image_path, key)):
#        a = os.path.join(image_path, key, x)
#        b = Image.open(a)
#        b.show()
#    return
#show_image_set("0A0A937C-016C-49E6-A9CA-480292B491BC")

