#!/usr/bin/env python
# coding: utf-8

# # 1. Load The Dataset

# In[1]:


#Import all the necessary libraries and print their versions
import sys
import nltk
import sklearn
import pandas
import numpy

print('Python: {}'.format(sys.version))
print('NLTK: {}'.format(nltk.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Numpy: {}'.format(numpy.__version__))


# In[2]:


# Load the Dataset
import pandas as pd
import numpy as np

data_source = "All_Review.xlsx"
df = pd.read_excel(data_source)


# In[3]:


#Print important information of the dataset
print(df.info())
print(df.head())


# In[4]:


#view the informations of datasets in pie and bar plots
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df.Restaurant.value_counts().plot(kind='pie', autopct='%1.0f%%')


# In[5]:


df.Positive_or_Negative.value_counts().plot(kind='pie', autopct='%1.0f%%')


# In[6]:


Positive_or_Negative = df.groupby(['Restaurant','Positive_or_Negative']).Positive_or_Negative.count().unstack()
Positive_or_Negative.plot(kind='bar')


# In[7]:


df.Real_or_Fake.value_counts().plot(kind='pie',autopct='%1.0f%%',colors=["red","yellow"])


# In[8]:


#check class distribution
classes = df.Real_or_Fake
print(classes.value_counts())


# # 2. Preprocess the Data

# In[9]:


#convert class labels to binary values, 0 = Fake, 1 = Real

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
Y = encoder.fit_transform(classes)

print(classes[:10])
print(Y[:10])


# In[10]:


#store the  Review datas
the_reviews = df.Reviews
print(the_reviews[:10])


# In[11]:


# use regular expressions to replace email addresses, URLs, phone numbers, other numbers

# Replace email addresses with 'email'
processed = the_reviews.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',
                                 'emailaddress')

# Replace URLs with 'webaddress'
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                  'webaddress')

# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
processed = processed.str.replace(r'£|\$', 'moneysymb')
    
# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                  'phonenumbr')
    
# Replace numbers with 'numbr'
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')


# In[12]:


# Remove punctuation
processed = processed.str.replace(r'[^\w\d\s]', ' ')

# Replace whitespace between terms with a single space
processed = processed.str.replace(r'\s+', ' ')

# Remove leading and trailing whitespace
processed = processed.str.replace(r'^\s+|\s+?$', '')


# In[13]:


# change words to lower case
processed = processed.str.lower()


# In[14]:


# remove stop words from text messages
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

processed = processed.apply(lambda x: ' '.join(
    term for term in x.split() if term not in stop_words))


# In[15]:


# Remove word stems using a Porter stemmer (ing, tenses, etc.)
ps = nltk.PorterStemmer()

processed = processed.apply(lambda x: ' '.join(
    ps.stem(term) for term in x.split()))


# In[16]:


print (processed)


# # 3. Generating Features

# In[17]:


# check the no. of most repeated words
from nltk.tokenize import word_tokenize

# create bag-of-words
all_words = []

for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)
        
all_words = nltk.FreqDist(all_words)


# In[18]:


# print the total number of words and the 15 most common words
print('Number of words: {}'.format(len(all_words)))
print('Most common words: {}'.format(all_words.most_common(15)))


# In[19]:


# use the 100 most common words as features
word_features = list(all_words.keys())[:100]


# In[20]:


# The find_features function will determine which of the 100 word features are contained in the review
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

# Lets see an example on first index
features = find_features(processed[0])
for key, value in features.items():
    if value == True:
        print (key)


# In[21]:


#printed om sentences
processed[0]


# In[22]:


#check if its in sentences or not
features


# In[23]:


# Now lets do it for all the reviews
messages = list(zip(processed, Y))

# define a seed for reproducibility
seed = 1
np.random.seed = seed
np.random.shuffle(messages)

# call find_features function for each review
featuresets = [(find_features(text), label) for (text, label) in messages]


# In[24]:


# we can split the featuresets into training and testing datasets using sklearn
from sklearn import model_selection

# split the data into training 75% and testing datasets 25%
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)


# In[25]:


print('Training:{}'.format(len(training)))
print('Testing:{}'.format(len(testing)))


# # 4. Scikit-Learn Classifiers with NLTK

# In[26]:


#import all classifiers to check which one has better accuracy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Define models to train
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = zip(names, classifiers)


#wrap models in NLTK
from nltk.classify.scikitlearn import SklearnClassifier
for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, testing)*100
    print("{} Accuracy: {}".format(name, accuracy))


# In[27]:


# Ensemble methods - Voting classifier
from sklearn.ensemble import VotingClassifier

names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = list((zip(names, classifiers)))

nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models, voting = 'hard', n_jobs = -1))
nltk_ensemble.train(training)
accuracy = nltk.classify.accuracy(nltk_model, testing)*100
print("Voting Classifier: Accuracy: {}".format(accuracy))


# In[28]:


# make class label prediction for testing set
txt_features, labels = list(zip(*testing))

prediction = nltk_ensemble.classify_many(txt_features)


# In[29]:


# print a confusion matrix and a classification report
print(classification_report(labels, prediction))

pd.DataFrame(
    confusion_matrix(labels, prediction),
    index = [['actual', 'actual'], ['Real', 'Fake']],
    columns = [['predicted', 'predicted'], ['Real', 'Fake']])


# In[ ]:




