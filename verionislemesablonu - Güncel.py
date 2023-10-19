# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 2023

"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#2.veri onisleme
#2.1.veri yukleme
kayitlarGuncel = pd.read_csv('VeriSetiTXT-Güncel.txt')


#kategorik -> numeric

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

Stres = kayitlarGuncel.iloc[:,6:7].values

ohe = preprocessing.OneHotEncoder()
#print(Stres)

Hazim = kayitlarGuncel.iloc[:,-1:].values
#print(Hazim)

Zaman = kayitlarGuncel.iloc[:,2:3].values
#print(Zaman)

Uyku = kayitlarGuncel.iloc[:,3:4].values
#print(Uyku)

Su = kayitlarGuncel.iloc[:,4:5].values
#print(Su)

Hareket = kayitlarGuncel.iloc[:,5:6].values
#print(Hareket)



sayisalVeriler = kayitlarGuncel.drop(['Besin','Tarih'], axis=1)

sayisalVeriler['Zaman'] = sayisalVeriler['Zaman'].str.replace('Sabah','0')
sayisalVeriler['Zaman'] = sayisalVeriler['Zaman'].str.replace('Ögle','1')
sayisalVeriler['Zaman'] = sayisalVeriler['Zaman'].str.replace('Aksam','2')

sayisalVeriler['Stres'] = sayisalVeriler['Stres'].str.replace('Az','0')
sayisalVeriler['Stres'] = sayisalVeriler['Stres'].str.replace('Orta','1')
sayisalVeriler['Stres'] = sayisalVeriler['Stres'].str.replace('çok','2')

sayisalVeriler['Hazim'] = sayisalVeriler['Hazim'].str.replace('Kötü','0')
sayisalVeriler['Hazim'] = sayisalVeriler['Hazim'].str.replace('Orta','1')
sayisalVeriler['Hazim'] = sayisalVeriler['Hazim'].str.replace('Iyi','2')


sayisalVeriler["Zaman"] = sayisalVeriler["Zaman"].astype(int)
sayisalVeriler["Stres"] = sayisalVeriler["Stres"].astype(int)
sayisalVeriler["Hazim"] = sayisalVeriler["Hazim"].astype(int)

#print(sayisalVeriler.dtypes)



'''
#histogram olusturarak veri analizi
for label in sayisalVeriler.columns[:-1]:
    plt.hist(sayisalVeriler[sayisalVeriler['Hazim']==0][label],color='red',label='kötü', alpha=1, density=True)
    plt.hist(sayisalVeriler[sayisalVeriler['Hazim']==1][label],color='orange',label='orta', alpha=0.7, density=True)
    plt.hist(sayisalVeriler[sayisalVeriler['Hazim']==2][label],color='green',label='iyi', alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("Olasılık")
    plt.xlabel(label)
    plt.legend()
    plt.show()
'''





#K-mean clustering
from sklearn.cluster import KMeans


#küme sayisinin belirlenmesi
def find_best_clusters(df, maximum_K):
    
    clusters_centers = []
    k_values = []
    
    for k in range(1, maximum_K):
        
        kmeans_model = KMeans(n_clusters = k, init='k-means++', n_init=1)
        kmeans_model.fit(df)
        
        clusters_centers.append(kmeans_model.inertia_)
        k_values.append(k)
        
    
    return clusters_centers, k_values


#elbow metodunun kullanilmasi
def generate_elbow_plot(clusters_centers, k_values):
    
    figure = plt.subplots(figsize = (12, 6))
    plt.plot(k_values, clusters_centers, 'o-', color = 'orange')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Cluster Inertia")
    plt.title("Elbow Plot of KMeans")
    plt.show()
    
clusters_centers, k_values = find_best_clusters(sayisalVeriler, 12)

generate_elbow_plot(clusters_centers, k_values)    




#1.Verilerin ölceklendirilmesi
scaledData = ((sayisalVeriler - sayisalVeriler.min())/(sayisalVeriler.max()-sayisalVeriler.min())) * 9 + 1
#print(scaledData.describe())


#2.Küme merkezlerinin rastgele belirlenmesi
def random_centroids(scaledData,k):
    centroids = []
    for i in range(k):
        centroid = scaledData.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis=1)

centroids = random_centroids(scaledData, 2)
#print(centroids)

#3. Verilerin her bir merkeze olan uzakliklarinin etiketlenmesi
def get_labels(scaledData, centroids):
    mesafeler = centroids.apply(lambda x: np.sqrt(((scaledData - x) **2).sum(axis=1)))
    #print(mesafeler)
    return mesafeler.idxmin(axis=1)

labels = get_labels(scaledData, centroids)
#print(labels.value_counts())


#4. Küme merkezlerinin güncellenmesi
def new_centroids(scaledData, labels, k):
    return scaledData.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T

#print(scaledData.shape)
from sklearn.decomposition import PCA
from IPython.display import clear_output


def plot_clusters(scaledData, labels, centroids, iteration):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(scaledData.iloc[:,:])
    centroids_2d = pca.transform(centroids.T)
    clear_output(wait=True)
    plt.title(f'Iterasyon {iteration}')
    plt.scatter(x=data_2d[:,0], y=data_2d[:,1], c=labels)
    plt.scatter(x=centroids_2d[:,0], y=centroids_2d[:,1])
    plt.show()
    

#5. 3. ve 4. adimlarin küme merkezleri 
# degismeyene kadar tekrar edilmesi
max_iteration = 50
k = 2 

centroids = random_centroids(scaledData, k)
old_centriods = pd.DataFrame()
iteration = 1 



while iteration < max_iteration and not centroids.equals(old_centriods):
    old_centriods = centroids
    
    labels = get_labels(scaledData, centroids)
    centroids = new_centroids(scaledData, labels, k)
    plot_clusters(scaledData, labels, centroids, iteration)
    iteration += 1
else:
    lastCentroids = centroids


#6. Listenin olusturulmasi
#liste = kayitlarGuncel[labels == 1]['Besin']

Besinler = kayitlarGuncel.iloc[:,1:2].values

Besinler = pd.DataFrame(data=Besinler, index = range(209), columns = ['Besinler'])
labels = pd.DataFrame(data=labels, index = range(209), columns = ['Label'])
Liste = pd.concat([Besinler,labels], axis=1)

etiketleme = pd.concat([kayitlarGuncel,labels], axis=1)

kötüBesinler = Liste.loc[Liste['Label']==1]
kötüBesinlerDrop = kötüBesinler.drop_duplicates()
kötüBesinlerDrop = kötüBesinler.drop_duplicates(keep='first')
print(kötüBesinlerDrop['Besinler'])






















####K-mean clustering####
from sklearn.cluster import KMeans

'''
#print(list(range(209)))

sonuc = pd.DataFrame(data=Zaman, index = range(209), columns = ['Zaman'])
#print(sonuc6)
sonuc1 = pd.DataFrame(data=Uyku, index = range(209), columns = ['Uyku'])
#print(sonuc3)
sonuc2 = pd.DataFrame(data=Su, index = range(209), columns = ['Su'])
#print(sonuc4)
sonuc3 = pd.DataFrame(data=Hareket, index = range(209), columns = ['Hareket'])
#print(sonuc5)
sonuc4 = pd.DataFrame(data=Stres, index = range(209), columns = ['Stres'])
#print(sonuc)
sonuc5 = pd.DataFrame(data=Hazim, index = range(209), columns = ['Hazim'])
#print(sonuc2)
sayisalVeriler = pd.concat([sonuc,sonuc1,sonuc2,sonuc3,sonuc4,sonuc5], axis=1)
#print(sayisalVeriler)
'''

'''
#trian, validation, test dataset
#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

#train, test = train_test_split(sayisalVeriler, test_size=0.33, random_state=0 )

#scaledData = ((train - train.min())/(train.max()-train.min())) * 9 + 1
#train, test = train_test_split(sayisalVeriler, test_size=0.33, random_state=0 )
#scaledData = ((train - train.min())/(train.max()-train.min())) * 9 + 1
#scaledTest = ((test - test.min())/(test.max()-test.min())) * 9 + 1
'''


'''
plt.scatter(sayisalVeriler["Uyku"],sayisalVeriler["Stres"])
plt.xlabel("Uyku")
plt.ylabel("Stres")
'''

'''
#verilerin ölçeklenmesi
scaler = StandardScaler()

scaler.fit(sayisalVeriler)

scaled_data = scaler.transform(sayisalVeriler)


#küme sayisinin belirlenmesi
def find_best_clusters(df, maximum_K):
    
    clusters_centers = []
    k_values = []
    
    for k in range(1, maximum_K):
        
        kmeans_model = KMeans(n_clusters = k, init='k-means++', n_init=1)
        kmeans_model.fit(df)
        
        clusters_centers.append(kmeans_model.inertia_)
        k_values.append(k)
        
    
    return clusters_centers, k_values

#elbow metodunun kullanilmasi
def generate_elbow_plot(clusters_centers, k_values):
    
    figure = plt.subplots(figsize = (12, 6))
    plt.plot(k_values, clusters_centers, 'o-', color = 'orange')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Cluster Inertia")
    plt.title("Elbow Plot of KMeans")
    plt.show()
    
#clusters_centers, k_values = find_best_clusters(sayisalVeriler, 12)

#generate_elbow_plot(clusters_centers, k_values)    



#train the model
kmeans_model = KMeans(n_clusters = 2, n_init=1)

kmeans_model.fit(scaled_data)

sayisalVeriler["clusters"] = kmeans_model.labels_
kayitlarGuncel["Clusters"] = kmeans_model.labels_
'''


'''
#cluster centers
#print(find_best_clusters(sayisalVeriler,3))
#print(kmeans_model.cluster_centers_)
print(kmeans_model.labels_)
'''

'''
#listeyi olusturma
Besinler = kayitlarGuncel.iloc[:,1:2].values
Clusters = kayitlarGuncel.iloc[:,-1].values

Besinler = pd.DataFrame(data=Besinler, index = range(209), columns = ['Besinler'])
Clusters = pd.DataFrame(data=Clusters, index = range(209), columns = ['Clusters'])
Liste = pd.concat([Besinler,Clusters], axis=1)

kötüBesinler = Liste.loc[Liste['Clusters']==0]
kötüBesinler = kötüBesinler.drop_duplicates()
kötüBesinler = kötüBesinler.drop_duplicates(keep='first')
print(kötüBesinler['Besinler'])
'''






'''
plt.scatter(sayisalVeriler["Zaman"], 
            sayisalVeriler["Hazim"], 
            c = sayisalVeriler["clusters"])
plt.xlabel("Zaman")
plt.ylabel("Hazim")
plt.show()

plt.scatter(sayisalVeriler["Stres"], 
            sayisalVeriler["Hazim"], 
            c = sayisalVeriler["clusters"])
plt.xlabel("Stres")
plt.ylabel("Hazim")
plt.show()

plt.scatter(sayisalVeriler["Hareket"], 
            sayisalVeriler["Hazim"], 
            c = sayisalVeriler["clusters"])
plt.xlabel("Hareket")
plt.ylabel("Hazim")
plt.show()
'''



















'''
#verilerin olceklenmesi
def scale_dataset(sayisalVerilerToplam, oversample=False):
    X = sayisalVerilerToplam[sayisalVerilerToplam.columns[:-1]].values
    y = sayisalVerilerToplam[sayisalVerilerToplam.columns[-1]].values
    
    sc=StandardScaler()
    #X = sc.fit_transform(X)
    
    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X, np.reshape(y, (-1,1))))
    
    return data, X, y

train, valid, test = np.split(sayisalVerilerToplam.sample(frac=1), [int(0.6*len(sayisalVerilerToplam)), int(0.8*len(sayisalVerilerToplam))])


train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)

'''









#train, test = train_test_split(sayisalVerilerToplam,test_size=0.33, random_state=0)


'''
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.transform(y_test)
'''




'''
#linear Regression
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,Y_train)

tahmin = regressor.predict(X_test)
'''

'''
#Grafiksel gösterim
#Hazim; kötü = 0 , orta = 1 , iyi = 2

from sklearn.datasets import make_blobs

sayisalVeriler, Hazim = make_blobs(n_samples=132, centers=3, random_state=1)
# summarize dataset shape
print(sayisalVeriler.shape, Hazim.shape)

# summarize observations by class label
from collections import Counter
counter = Counter(Hazim)
print(counter)

# summarize first few examples
for i in range(132):
 print(sayisalVeriler[i], Hazim[i])
 
# plot the dataset and color the by class label
from numpy import where
from matplotlib import pyplot

for label, _ in counter.items():
 row_ix = where(Hazim == label)[0]
 pyplot.scatter(sayisalVeriler[row_ix, 0], sayisalVeriler[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
'''



'''
for label in sayisalVerilerToplam.columns[1:]:
    plt.scatter(sayisalVerilerToplam[label], sayisalVerilerToplam["kötü"])
    plt.title(label)
    plt.ylabel("kötü hazim")
    plt.xlabel(label)
    plt.show()
'''

#kayitlar["Stres"]=(kayitlar["Stres"]=="Orta").astype(int)







'''
ulke = kayitlar.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(kayitlar.iloc[:,0])
print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

#encoder: kategorik -> numeric
c = kayitlar.iloc[:,-1:].values
print(c)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

c[:,-1] = le.fit_transform(kayitlar.iloc[:,-1])
print(c)

ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)

print(list(range(22)))
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = kayitlar.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c[:,:1], index = range(22), columns = ['cinsiyet'])
print(sonuc3)

s=pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2=pd.concat([s,sonuc3], axis=1)
print(s2)


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)



from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

boy = s2.iloc[:,3:4].values
print(boy)

sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)

x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)

r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred = r2.predict(x_test)



import statsmodels.api as sm

X = np.append(arr=np.ones((22,1)).astype(int), values=veri, axis=1)

X_l = veri.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())

'''








