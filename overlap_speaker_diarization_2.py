from __future__ import division
import librosa as li
import numpy as np 
from sklearn.cluster import AffinityPropagation, KMeans
from scipy import stats
from sklearn.mixture import GMM
from math import *
from scipy.fftpack import fft
file_name = r"C:\\Python27\\rbang.wav"
audio_time_series, sample_rate = li.load(r"C:\\Python27\\spkr2.wav")
length_series = len(audio_time_series)
print(length_series)
zero_crossings = []
energy = []
entropy_of_energy = []
mfcc = []
pitch=[]
chroma_stft = []
res=[]
temp=[]
thresh=0.451
stored_series=[]
pitch=[]
for i in range(0,length_series,int(sample_rate/5.0)):
     frame_self = audio_time_series[i:i+int(sample_rate/5.0):1]
     stored_series.append(frame_self)
     mt = []
     mf = li.feature.mfcc(frame_self)
     for k in range(0,len(mf)):
         mt.append(np.mean(mf[k]))
     mfcc.append(mt)
     e = li.feature.rmse(frame_self)
     energy.append(np.mean(e))
     ct = []
     cf = li.feature.chroma_stft(frame_self)
     for k in range(0,len(cf)):
          ct.append(np.mean(cf[k]))
     ca=sum(ct)/len(ct)
     chroma_stft.append(ca)
     pitches, magnitudes = li.piptrack(y=frame_self, sr=sample_rate)
     pitch.append(pitches[0][0])
     onset_env = li.onset.onset_strength(frame_self, sr=sample_rate)
     tempo = li.beat.tempo(onset_envelope=onset_env, sr=sample_rate)
     temp.append(tempo[0])
f_list_1 = []
f_list_1.append(energy)
f_np_1 = np.array(f_list_1)
f_np_1 = np.transpose(f_np_1)
f_np_3 = np.array(mfcc)
master = np.concatenate([f_np_1,f_np_3], axis=1)
sp_centroid = []
sp_bandwidth = []
sp_contrast = []
sp_rolloff = []
for i in range(0,length_series,int(sample_rate/5.0)):
     frame_self = audio_time_series[i:i+int(sample_rate/5.0):1]
     cp = li.feature.spectral_centroid(y=frame_self, hop_length=220500)
     sp_centroid.append(cp[0][0])
     bp = li.feature.spectral_bandwidth(y=frame_self, hop_length=220500)
     sp_bandwidth.append(bp[0][0])
     csp = li.feature.spectral_contrast(y=frame_self, hop_length=220500)
     sp_contrast.append(np.mean(csp))
     rsp = li.feature.spectral_rolloff(y=frame_self, hop_length=220500)
     sp_rolloff.append(np.mean(rsp[0][0]))
f_list_2 = []
f_list_2.append(sp_centroid)
f_list_2.append(sp_bandwidth)
f_list_2.append(sp_contrast)
f_list_2.append(sp_rolloff)
f_np_2 = np.array(f_list_2)
f_np_2 = np.transpose(f_np_2)

gmm = GMM(n_components=2).fit(master)
probs = gmm.predict_proba(master)
print("Probability matrix:\n"+str(probs))
overlap_res=[int() for  x in range(len(probs))]
for  x in range(len(probs)):
    prob1=probs[x][0]
    prob2=probs[x][1]
    logval=log(prob1/prob2)
    if(logval>=thresh):
        overlap_res[x]=1
    else:
        overlap_res[x]=0
print("Overlapping result:\n"+str(overlap_res))
single_feature=[]
sind=[]
for i in range(len(overlap_res)):
    if(overlap_res[i]==0):
        single_feature.append(master[i])
        sind.append(i)
gmm_das = GMM(n_components=2).fit(single_feature)
probs_das = gmm_das.predict_proba(single_feature)
print(probs_das)
spk0,spk1=0,0
Y=[] 
ftr1,ftr2=[],[]
ats1,ats2=[],[]
for i in range(len(probs_das)):
    if(probs_das[i][0]>probs_das[i][1]):
        print("At Segment: "+str(sind[i])+" speaker 0 is speaking")
        ats1.extend(stored_series[sind[i]])
        Y.append(0)
        spk0+=1
        ftr1.append(single_feature[i])
    else:
        print("At Segment: "+str(sind[i])+" speaker 1 is speaking")
        Y.append(1)
        ats2.extend(stored_series[sind[i]])
        spk1+=1
        ftr2.append(single_feature[i])
print("Individually speaker 0 spoke "+str(spk0)+" times and speaker 1 spoke "+str(spk1)+" times.")
Y=np.array(Y)
'''
single_feature=np.array(single_feature)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import array
from keras.models import load_model
'''
from sklearn.neural_network import MLPClassifier as mlp
'''
'''
'''
model = Sequential()a,b=single_feature.shape
single_feature = np.reshape(single_feature, (single_feature.shape[0], 1, single_feature.shape[1]))
model.add(LSTM(10, input_shape=(single_feature.shape[1],single_feature.shape[2])))
model.add(Dense(1, activation='linear'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(single_feature, Y, epochs=300, shuffle=False, verbose=0)
'''
clf = mlp(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(single_feature,Y)
for i in range(len(overlap_res)):
     if(overlap_res[i]==1):
          result=clf.predict(np.array([master[i]]))
          print("At segment "+str(i)+" speaker "+ str(result) +" spoke more than speaker "+ str(abs(1-result)))
gmm2 = GMM(n_components=2).fit(ftr1)
probs2 = gmm2.predict_proba(ftr1)
gmm3 = GMM(n_components=2).fit(ftr2)
probs3 = gmm3.predict_proba(ftr2)
print("Probability prediction for dialect of speaker 0:\n"+str(probs2))
print("Probability prediction for dialect of speaker 1:\n"+str(probs3))
prob_spk_1u,prob_spk_2u,prob_spk_1i,prob_spk_2i=0.0,0.0,0.0,0.0
_prob_spk_1u,_prob_spk_2u,_prob_spk_1i,_prob_spk_2i=0.0,0.0,0.0,0.0
for i in range(0,len(probs2)):
     prob_spk_1u+=probs2[i][0]
     if(probs2[i][0]!=0):
          prob_spk_1i*=probs2[i][0]
     _prob_spk_1u+=probs2[i][1]
     if(probs2[i][1]!=0):
          _prob_spk_1i*=probs2[i][1]
for i in range(0,len(probs3)):
     prob_spk_2u+=probs3[i][0]
     if(probs3[i][0]!=0.0):
          prob_spk_2i*=probs3[i][0]
     _prob_spk_2u+=probs3[i][1]
     if(probs3[i][1]!=0.0):
          _prob_spk_2i*=probs3[i][1]
spk1res1,spkres2=prob_spk_1u,prob_spk_2u
_spk1res1,_spkres2=_prob_spk_1u,_prob_spk_2u
dialect1,dialect2=None,None
if(spk1res1<_spk1res1):
     dialect1="Indian"
else:
     dialect1="American"
if(spkres2<_spkres2):
     dialect2="Indian"
else:
     dialect2="American"
print("Speaker 0 is of dialect: "+dialect1+" and Speaker 1 is of dialect: "+dialect2)
if(dialect1==dialect2):
     print("Speakers may be from same region")
else:
     print("Speakers may be from different region")
f_list_1 = []
f_list_1.append(energy)
f_np_1 = np.array(f_list_1)
f_np_1 = np.transpose(f_np_1)
master = np.concatenate([f_np_1,f_np_3], axis=1)
single_feature=[]
for i in range(len(overlap_res)):
    if(overlap_res[i]==0):
        single_feature.append(master[i])
ftr1,ftr2=[],[]
for i in range(len(probs_das)):
    if(probs_das[i][0]>probs_das[i][1]):
        ftr1.append(single_feature[i])
    else:
        ftr2.append(single_feature[i])
from sklearn.svm import SVC
from collections import defaultdict
agedict={1:"Below 20",0:"Young",2:"Senior Citizen"}
ageftr1=[list(ftr1[x]) for x in range(len(ftr1)) if x%2==0]
cluster_obj1 = KMeans(n_clusters = 3,random_state=0).fit(ageftr1)
res1 = cluster_obj1.predict(ageftr1)
clf=SVC()
clf.fit(ageftr1,res1)
problist=[]
for x in range(len(ftr1)):
    if(x%2!=0):
        problist.append(clf.predict([list(ftr1[x])])[0])
d = defaultdict(int)
for i in problist:
    d[i] += 1
result = max(d.iteritems(), key=lambda x: x[1])
print(agedict[result[0]])
ageftr2=[list(ftr2[x]) for x in range(len(ftr2)) if x%2==0]
cluster_obj2 = KMeans(n_clusters = 3 ,random_state=0).fit(ageftr2)
res2 = cluster_obj2.predict(ageftr2)
clf=SVC()
clf.fit(ageftr2,res2)
problist=[]
for x in range(len(ftr2)):
    if(x%2!=0):
        problist.append(clf.predict([list(ftr2[x])])[0])
d = defaultdict(int)
for i in problist:
    d[i] += 1
result = max(d.iteritems(), key=lambda x: x[1])
print(agedict[result[0]])
f_list_1 = []
f_list_1.append(energy)
f_list_1.append(chroma_stft)
f_np_1 = np.array(f_list_1)
f_np_1 = np.transpose(f_np_1)
master = np.concatenate([f_np_1,f_np_3], axis=1)
single_feature=[]
for i in range(len(overlap_res)):
    if(overlap_res[i]==0):
        single_feature.append(master[i])
ftr1,ftr2=[],[]
for i in range(len(probs_das)):
    if(probs_das[i][0]>probs_das[i][1]):
        ftr1.append(single_feature[i])
    else:
        ftr2.append(single_feature[i])
agedict={0:"Angry",1:"Happy",2:"Sad",3:"Neutral"}
ageftr1=[list(ftr1[x]) for x in range(len(ftr1)) if x%2==0]
cluster_obj1 = KMeans(n_clusters = 4 ,random_state=0).fit(ageftr1)
res1 = cluster_obj1.predict(ageftr1)
clf=SVC()
clf.fit(ageftr1,res1)
problist=[]
for x in range(len(ftr1)):
    if(x%2!=0):
        problist.append(clf.predict([list(ftr1[x])])[0])
d = defaultdict(int)
for i in problist:
    d[i] += 1
result = max(d.iteritems(), key=lambda x: x[1])
i=-1
for x in problist:
    if(x!=result[0]):
        i=x
        break
if(result[0]==0):
    if(result[1]==1 or result[1]==3):
        print("Passively Angry")
    else:
        print("Actively Angry")
elif(result[0]==2):
    if(i==1 or i==3):
        print("Passively Angry")
    else:
        print("Actively Angry")
elif(result[0]==1):
    print("Happy")
else:
    print("Neutral")
ageftr2=[list(ftr2[x]) for x in range(len(ftr2)) if x%2==0]
cluster_obj2 = KMeans(n_clusters = 4 ,random_state=0).fit(ageftr2)
res2 = cluster_obj2.predict(ageftr2)
clf=SVC()
clf.fit(ageftr2,res2)
problist=[]
for x in range(len(ftr2)):
    if(x%2!=0):
        problist.append(clf.predict([list(ftr2[x])])[0])
d = defaultdict(int)
for i in problist:
    d[i] += 1
result = max(d.iteritems(), key=lambda x: x[1])
i=-1
for x in problist:
    if(x!=result[0]):
        i=x
        break
if(result[0]==0):
    if(i==1 or i==3):
        print("Passively Angry")
    else:
        print("Actively Angry")
elif(result[1]==2):
    if(result[1]==1 or result[1]==3):
        print("Passively Angry")
    else:
        print("Actively Angry")
else:
    print("Happy")
import numpy as np
from scipy.io.wavfile import write
write('test1.wav', sample_rate, np.array(ats1))
write('test2.wav', sample_rate, np.array(ats2))

f_list_1 = []
f_list_1.append(pitch)
f_list_1.append(temp)
f_np_1 = np.array(f_list_1)
f_np_1 = np.transpose(f_np_1)
master = np.concatenate([f_np_1,f_np_2], axis=1)
single_feature=[]
for i in range(len(overlap_res)):
    if(overlap_res[i]==0):
        single_feature.append(master[i])
ftr1,ftr2=[],[]
for i in range(len(probs_das)):
    if(probs_das[i][0]>probs_das[i][1]):
        ftr1.append(single_feature[i])
    else:
        ftr2.append(single_feature[i])
agedict={0:"Male",1:"Female"}
ageftr1=[list(ftr1[x]) for x in range(len(ftr1)) if x%2==0]
cluster_obj1 = KMeans(n_clusters = 2 ,random_state=0).fit(ageftr1)
res1 = cluster_obj1.predict(ageftr1)
clf=SVC()
clf.fit(ageftr1,res1)
problist=[]
for x in range(len(ftr1)):
    if(x%2!=0):
        problist.append(clf.predict([list(ftr1[x])])[0])
d = defaultdict(int)
for i in problist:
    d[i] += 1
result = max(d.iteritems(), key=lambda x: x[1])
print(agedict[result[0]])
ageftr2=[list(ftr2[x]) for x in range(len(ftr2)) if x%2==0]
cluster_obj2 = KMeans(n_clusters = 2 ,random_state=0).fit(ageftr2)
res2 = cluster_obj2.predict(ageftr2)
clf=SVC()
clf.fit(ageftr2,res2)
problist=[]
for x in range(len(ftr2)):
    if(x%2!=0):
        problist.append(clf.predict([list(ftr2[x])])[0])
d = defaultdict(int)
for i in problist:
    d[i] += 1
result = max(d.iteritems(), key=lambda x: x[1])
print(agedict[result[0]])
