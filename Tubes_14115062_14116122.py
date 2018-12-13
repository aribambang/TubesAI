#!/usr/bin/env python
# coding: utf-8

# # Tugas Besar IF3170 Artificial Intelligence<br>Teknik Informatika<br>Institut Teknologi Sumatera
# 
# #### ARI BAMBANG KURNIAWAN (14115062) <br> MARIA OKTARISE N. G. (14116122)
# 
# ## Inisialisasi library

# In[1]:


#library untuk proses
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

#library untuk pembelajaran
import pandas as pd
from sklearn import datasets, metrics, neighbors, tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
import graphviz 
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib


# ## inisialisasi algoritma

# In[2]:


gnb = GaussianNB()
dt = tree.DecisionTreeClassifier()
knn = neighbors.KNeighborsClassifier(5, weights='uniform')
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)


# ## A. Membaca dataset standar iris dan dataset play-tennis (dataset eksternal dalam format csv).
# ### A.1 Membaca dataset Iris

# In[3]:


df = datasets.load_iris()


# In[4]:


df.feature_names


# In[5]:


df.data


# In[6]:


df.target


# ### A.2 Membaca dataset play-tennis

# In[7]:


df1 = pd.read_csv('tennis.csv')
df1.head()


# ## B. Melakukan pembelajaran NaiveBayes, DecisionTree ID3, kNN, dan Neural Network MLP untuk dataset iris dengan skema full-training, dan menampilkan modelnya. 
# 
# ### B.1 Naive Bayes

# In[8]:


iris = gnb.fit(df.data, df.target)
print("Model: ")
print("")
print("Probabilitas setiap kelas: ")
print(iris.class_prior_)
print("")
print("Rata fitur per kelas: ")
print(iris.theta_)
print("")
print("Probabilitas tiap fitur bila diberikan kelas: ")
print(iris.sigma_)


# ### B.2 Decision Tree

# In[9]:


dt_model = dt.fit(df.data, df.target)

dot_data = tree.export_graphviz(dt_model, out_file=None, 
                         feature_names=df.feature_names,  
                         class_names=df.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)

graph = graphviz.Source(dot_data)


# In[10]:


graph.render("iris_dt")


# In[11]:


graph


# ### B.3 k-Nearest Neighbor (kNN)
# 
# kNN tidak menghasilkan model

# In[12]:


knn.fit(df.data, df.target)


# ### B.5 Neural Network MLP

# In[13]:


mlp_model = mlp.fit(df.data, df.target)
mlp_model


# ## C. Melakukan pembelajaran NaïveBayes, DecisionTree, kNN, dan MLP untuk dataset iris dengan skema split train 90% dan test 10%, dan menampilkan kinerja serta confusion matrixnya.
# 
# ### Membuat dataset iris menjadi 10% test

# In[14]:


X_train, X_test, y_train, y_test= train_test_split(df.data, df.target, test_size=0.1)


# ### C.1 Naive Bayes

# In[15]:


bayes_model = gnb.fit(X_train, y_train)
prediction_bayes = bayes_model.predict(X_test)
precision_bayes = metrics.precision_score(y_test, prediction_bayes, average=None) * 100
recall_bayes = metrics.recall_score(y_test, prediction_bayes, average=None) * 100

print("Measurement prediction: \n")
print("1. Accuration: %f" % (np.mean(prediction_bayes == y_test) * 100) + "%\n")
print("2. Precision:")
for i in range(3):
    print(df.target_names[i] + ": " + str(precision_bayes[i]) + "%")
print("")
print("3. Recall:")
for i in range(3):
    print(df.target_names[i] + ": " + str(recall_bayes[i]) + "%")
print("")

print("")
print("Confusion Matrix: ")
print(metrics.confusion_matrix(y_test, prediction_bayes))


# ### C2. Decision Tree

# In[16]:


dt_model = dt.fit(X_train, y_train)
prediction_dt = dt_model.predict(X_test)
precision_dt = metrics.precision_score(y_test, prediction_dt, average=None) * 100
recall_dt = metrics.recall_score(y_test, prediction_dt, average=None) * 100

print("Measurement prediction: \n")
print("1. Accuration: %f" % (np.mean(prediction_dt == y_test) * 100) + "%\n")
print("2. Precision:")
for i in range(3):
    print(df.target_names[i] + ": " + str(precision_dt[i]) + "%")
print("")
print("3. Recall:")
for i in range(3):
    print(df.target_names[i] + ": " + str(recall_dt[i]) + "%")
print("")

print("")
print("Confusion Matrix: ")
print(metrics.confusion_matrix(y_test, prediction_dt))


# ### C.3 kNN

# In[17]:


knn.fit(X_train, y_train)
prediction_knn = knn.predict(X_test)
precision_knn = metrics.precision_score(y_test, prediction_knn, average=None) * 100
recall_knn = metrics.recall_score(y_test, prediction_knn, average=None) * 100

print("Measurement prediction: \n")
print("1. Accuration: %f" % (np.mean(prediction_knn == y_test) * 100) + "%\n")
print("2. Precision:")
for i in range(3):
    print(df.target_names[i] + ": " + str(precision_knn[i]) + "%")
print("")
print("3. Recall:")
for i in range(3):
    print(df.target_names[i] + ": " + str(recall_knn[i]) + "%")
print("")

print("")
print("Confusion Matrix: ")
print(metrics.confusion_matrix(y_test, prediction_knn))


# ### C.4 Neural Network MLP

# In[18]:


mlp_model = mlp.fit(X_train, y_train)
prediction_mlp = mlp_model.predict(X_test)
precision_mlp = metrics.precision_score(y_test, prediction_mlp, average=None) * 100
recall_mlp = metrics.recall_score(y_test, prediction_mlp, average=None) * 100

print("Measurement prediction: \n")
print("1. Accuration: %f" % (np.mean(prediction_mlp == y_test) * 100) + "%\n")
print("2. Precision:")
for i in range(3):
    print(df.target_names[i] + ": " + str(precision_mlp[i]) + "%")
print("")
print("3. Recall:")
for i in range(3):
    print(df.target_names[i] + ": " + str(recall_mlp[i]) + "%")
print("")

print("")
print("Confusion Matrix: ")
print(metrics.confusion_matrix(y_test, prediction_mlp))
print("")
print("Note: error dibawah adalah kinerja dengan dataset Iris tidak optimal pada Neural Network karena harus lebih banyak data training")


# ### D. Melakukan pembelajaran NaïveBayes, DecisionTree, kNN, dan MLP untuk dataset iris dengan skema 10-fold cross validation, dan menampilkan kinerjanya.
# 
# ### D.1 Naive Bayes

# In[19]:


scores_bayes = cross_val_score(gnb, df.data, df.target, cv=10)

i = 1
print("Kinerja Naive Bayes: \n")
for score in scores_bayes:
    print("Fold %d" % i + " = %f" % score)
    i+=1
print
print("\nRata-rata = %f"% np.mean(scores_bayes))


# ### D.2 Decision Tree

# In[20]:


scores_dt = cross_val_score(dt, df.data, df.target, cv=10)

i = 1
print("Kinerja Decision Tree: \n")
for score in scores_dt:
    print("Fold %d" % i + " = %f" % score)
    i+=1
print
print("\nRata-rata = %f"% np.mean(scores_dt))


# ### D.3 kNN

# In[21]:


scores_knn = cross_val_score(knn, df.data, df.target, cv=10)

i = 1
print("Kinerja kNN: \n")
for score in scores_knn:
    print("Fold %d" % i + " = %f" % score)
    i+=1
print
print("\nRata-rata = %f"% np.mean(scores_knn))


# ### D.4 Neural Network MLP

# In[22]:


scores_mlp = cross_val_score(mlp, df.data, df.target, cv=10)

i = 1
print("Kinerja MLP: \n")
for score in scores_mlp:
    print("Fold %d" % i + " = %f" % score)
    i+=1
print
print("\nRata-rata = %f"% np.mean(scores_mlp))


# ## E. Menyimpan (save) model/hipotesis hasil pembelajaran ke sebuah file eksternal 
# 
# ### E.1 Naive Bayes

# In[23]:


joblib.dump(gnb, 'iris_gnb.mdl')


# ### E.2 Decision Tree

# In[24]:


joblib.dump(dt, 'iris_dt.mdl')


# ### E.3 kNN

# In[25]:


joblib.dump(knn, 'iris_knn.mdl')


# ### E.4 Neural Network MLP

# In[26]:


joblib.dump(mlp, 'iris_mlp.mdl')


# ## F. Membaca (read)model/hipotesis dari file eksternal
# 
# ### F.1 Naive Bayes

# In[27]:


gnb = joblib.load('iris_gnb.mdl')


# ### F.2 Decision Tree

# In[28]:


dtl = joblib.load('iris_dt.mdl')


# ### F.3 kNN

# In[29]:


kNN = joblib.load('iris_knn.mdl')


# ### F.4 Neural Network MLP

# In[30]:


mlp = joblib.load('iris_mlp.mdl')


# ## G. Membuat instance baru dengan memberi nilai untuk setiap atribut

# In[31]:


atribut_baru = [3.4, 3.3, 2.4, 0.5]
instance_baru = np.array([atribut_baru])
                
print("instance baru:")
for i in range(4):
    print(df.feature_names[i] + ":", instance_baru[0][i])


# ## H. Melakukan klasifikasi dengan memanfaatkan model/hipotesis NaïveBayes, DecisionTree, dan kNN dan instance pada g.  
# 
# ### H.1 Naive Bayes

# In[32]:


print(df.target_names[gnb.predict(instance_baru)])


# ### H.2 Decision Tree

# In[33]:


print(df.target_names[dt.predict(instance_baru)])


# ### H.3 kNN

# In[34]:


print(df.target_names[knn.predict(instance_baru)])


# ### H.4 Neural Network MLP

# In[35]:


print(df.target_names[mlp.predict(instance_baru)])

