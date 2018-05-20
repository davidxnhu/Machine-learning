#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler
from sklearn import cross_validation

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV

import numpy as np

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Explore the dataset

def exploration(data_dict):
    '''
    Given a dictionary, pring the number of keys, the number of features,
    an example data point, 
    number and percentage of poi
    '''

    total_people = len(data_dict)
    print 'Total number of people in the dataset:',total_people
    print 'POI is the label, and total number of features in the dataset:', \
          len(data_dict['METTS MARK'])-1
    print 'An example of the dataset:\n',data_dict['METTS MARK']

    total_poi=sum(data_dict[d]['poi'] for d in data_dict)
    print 'Total number of POI in the dataset:', \
          total_poi,'\nTotal percentage of POI in the dataset',\
          total_poi*1.0/total_people

exploration(data_dict)

### Task 1: Remove outliers

def find_outlier(data_dict):
    features = ['poi','salary','bonus']
    data = featureFormat(data_dict, features) ##data_format by extract data to numpy array
    labels, features = targetFeatureSplit(data)

    #visualize the data and find 1 outlier with extremyly high income noticed
    for point in features:
        salary = point[0]
        bonus = point[1]
        plt.scatter(salary, bonus)

    plt.xlabel("salary")
    plt.ylabel("bonus")
    plt.show()

find_outlier(data_dict)

# find out and remove this outlier

for d in data_dict:
    if data_dict[d]['salary']!='NaN' and data_dict[d]['salary']>2.5e7:
        print d

# if 'NaN is not excluded, there will be a lot of names; after removal, the only name is 'TOTAL'

data_dict.pop('TOTAL',0)

# repeat the process again, there are 4 more outliers which are not very far.
find_outlier(data_dict)

#print sorted(data_dict.keys())

# find out there is a name 'THE TRAVEL AGENCY IN THE PARK',
# which should be removed too

data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)

# Explore the data_set again
exploration(data_dict)


    
### Task 2: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### REMOVE FEATURES
### From the pdf document, 'total_payments' and 'total_stock_value'
### are linear sum of other variables. I decide to remove them from
### the features list. 
### 'email_address' is a text string and corresponds to the person's name,
### I decide to remove it from the features list as well. 

### CREATE NEW FEATURES
'''
from_poi_to_this_person_ratio = from_poi_to_this_person/to_messages
from_this_person_to_poi_ratio = from_this_person_to_poi/from_messages
'''

for d in data_dict:
    if data_dict[d]['from_poi_to_this_person'] != 'NaN':
        data_dict[d]['from_poi_to_this_person_ratio'] = 1.*data_dict[d]['from_poi_to_this_person']/data_dict[d]['to_messages']
    else:
        data_dict[d]['from_poi_to_this_person_ratio'] = 'NaN'


    if data_dict[d]['from_this_person_to_poi'] != 'NaN':
        data_dict[d]['from_this_person_to_poi_ratio'] = 1.*data_dict[d]['from_this_person_to_poi']/data_dict[d]['from_messages']
    else:
        data_dict[d]['from_this_person_to_poi_ratio'] = 'NaN'



### Extract features and labels from dataset for local testing
features_list  = ['poi','salary', 'deferral_payments', 'loan_advances', \
                  'bonus', 'restricted_stock_deferred', 'deferred_income', \
                  'expenses', 'exercised_stock_options', 'other', \
                  'long_term_incentive', 'restricted_stock', 'director_fees',\
                  'to_messages', 'from_poi_to_this_person', 'from_messages', \
                  'from_this_person_to_poi', 'shared_receipt_with_poi',\
                  'from_poi_to_this_person_ratio',\
                  'from_this_person_to_poi_ratio']    

data = featureFormat(data_dict, features_list,sort_keys=False) ##data_format by extract data to numpy array
labels, features = targetFeatureSplit(data)


### split the data into train and test

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)


###Feature scaling before selection, necessary for SVM. 
scaler = MinMaxScaler()
rescaled_features_train = scaler.fit_transform(features_train)
rescaled_features_test = scaler.fit_transform(features_test)

from sklearn.feature_selection import SelectKBest 
selection = SelectKBest(k=1)

from sklearn.pipeline import Pipeline, FeatureUnion
combined_features = FeatureUnion([("univ_select", selection)])

features_transformed = selection.fit(rescaled_features_train, labels_train).transform(rescaled_features_train)

svm=SVC(kernel='linear')

pipeline = Pipeline([("features", combined_features), ("svm", svm)])

param_grid = dict(features__univ_select__k=[1, 2, 4, 6],
                  svm__C=[1,10,1e2,1e3])

grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv = 5, verbose=10)
grid_search.fit(rescaled_features_train, labels_train) ###only use train set
print(grid_search.best_estimator_)


###SelectKBest score
selector2 = SelectKBest(k=1)
selector2.fit(rescaled_features_train, labels_train)
scores =  selector2.scores_
print scores


### distribution of score in selectkbest
plt.hist(scores,bins = 19)
plt.title("SelectKBest score disctribution")
plt.xlabel("Score")
plt.ylabel("Count")
plt.show()


features_score=[]
for i in range(len(scores)):
    features_score.append((features_list[i+1],scores[i]))

features_score=sorted(features_score,key=lambda x:x[1],reverse=True)

print features_score



### Final feature selection
features_list = ['poi', 'exercised_stock_options', \
                 'from_this_person_to_poi_ratio', 'expenses','salary']


### find NaN features for a given variable

def qualify(data,variable):
    '''
    in a dictionary, for the key variable, return the number of NaN value
    '''
    count=0
    for d in data:
        if data[d][variable]=='NaN':
            count+=1
    return count,1.0*count/len(data)

for f in features_list:
    print 'number and percentage of missing values for',f,':',qualify(data_dict,f)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

### at this moment, the data has been updated with outlier removal and new features added
my_dataset = data_dict

data = featureFormat(my_dataset, features_list, sort_keys = False)


### plot every pair of variables in the features_list

import itertools


for i1,i2 in itertools.combinations(range(len(features_list)),2):
    for e in data:
        poi=e[0]
        if abs(poi-0.0)<0.001: # person is poi
            point_color='b'
        else:
            point_color='r'

        plt.scatter(e[i1],e[i2],color=point_color)

    plt.xlabel(features_list[i1])
    plt.ylabel(features_list[i2])
    plt.show()




### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()


labels,features=targetFeatureSplit(data)

names=['Naive Bayes','Linear SVM', 'RBF SVM','Decision Tree']
classifiers=[GaussianNB(),SVC(kernel='linear',C=10),\
             SVC(kernel='rbf',gamma=0.1,C=1000),\
             DecisionTreeClassifier()]


features = np.array(features)
labels = np.array(labels)

def performance(clf,features,labels):
    '''
    split features,labels using StratifiedShuffleSplit and
    calculate the metrics: accuracy,precision,and recall

    '''
    cv=cross_validation.StratifiedShuffleSplit(labels,1000,random_state=0) #test_size=0.1
    true_negatives,false_negatives,true_positives,false_positives=0,0,0,0

    for train_index,test_index in cv:
        features_train,features_test=features[train_index],features[test_index]
        labels_train,labels_test=labels[train_index],labels[test_index]

    
        scaler=MinMaxScaler()
        rescaled_features_train=scaler.fit_transform(features_train)
        rescaled_features_test=scaler.fit_transform(features_test)

        ### fit the classifier using training set
        clf.fit(rescaled_features_train,labels_train)
        predictions=clf.predict(rescaled_features_test)

        for pred,truth in zip(predictions,labels_test):
            if pred==0 and truth==0:
                true_negatives+=1
            elif pred==0 and truth ==1:
                false_negatives+=1
            elif pred==1 and truth==0:
                false_positives+=1
            else:
                true_positives+=1

    try:
        total=true_negatives+false_negatives+false_positives+true_positives
        accuracy=(true_negatives+true_positives)*1.0/total
        precision=true_positives*1.0/(true_positives+false_positives)
        recall=true_positives*1.0/(true_positives+false_negatives)
        f1=2.0*true_positives/(2*true_positives+false_positives+false_negatives)
        f2=(1+2.0*2.0)*precision*recall/(4*precision+recall)
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."


for name,clf in zip(names,classifiers):
    print '###############'
    print
    print name
    print
    print '###############'
    performance(clf,features,labels)
    

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)



parameters={'kernel':['rbf'],'C':[1,10,100,1000,10000],'gamma':[0.001,0.01,0.1,1]}

for k in parameters['kernel']:
    for c in parameters['C']:
        for g in parameters['gamma']:
            clf=SVC(kernel=k,C=c,gamma=g)

            print 'kernel',k,'C',c,'gamma',g
            performance(clf,features,labels)


clf=SVC(C=10000, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)


### without new feature
features_list2=['poi', 'exercised_stock_options', \
                  'expenses','salary']

data2=featureFormat(my_dataset,features_list2)
labels2,features2=targetFeatureSplit(data2)
labels2,features2=np.array(labels2),np.array(features2)

performance(clf,features2,labels2)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

