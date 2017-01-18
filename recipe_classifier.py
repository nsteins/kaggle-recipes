import pandas as pd
import numpy as np
from collections import Counter
import operator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, ClassifierMixin
import pdb

        

class UniqueItemClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        total = Counter(','.join(X).split(','))
        total_tot = len(total)
        self.item_dicts_ = dict.fromkeys(y.unique())
        for category in y.unique():
            ing = ','.join(X[y==category]).split(',')
            ing_cts = Counter(ing)
            ing_tot = len(ing_cts)
            ing_score = {k:(float(ing_cts[k])/ing_tot)/(float(total[k])/total_tot) 
                        for k in ing_cts.viewkeys()}   
            self.item_dicts_[category] = ing_score
            ing_score = ing_score.items()
            #print category
            #print sorted(ing_score, key = lambda x:x[1],reverse=True)[0:10]
        return self

    def predict(self, X, probability=False):
        out = np.empty(X.shape[0],dtype='object')
        p = []
        for i,item in enumerate(list(X)):
            ing = item.split(',')
            score = dict.fromkeys(self.item_dicts_.keys())
            for cat,score_dict in self.item_dicts_.iteritems():
                cat_score = 0
                for item in ing:
                    try:
                        cat_score += score_dict[item]
                    except KeyError:
                        pass
                score[cat] = cat_score
            out[i] = max(score.iteritems(), key=operator.itemgetter(1))[0]
            p.append(score)
        if probability:
            return out,p
        else:
            return out

class NestedLogisticRegressionClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self, categories = [],n_categories = 4):
        self.categories = categories
        self.n_categories = n_categories
        
    def fit(self, X, y):
        if self.categories:
            catval = pd.Series(data=-1,index = y.index)
            for i, category in enumerate(self.categories):
                ind = y.isin(category)
                catval[ind] = i
        else:
            km = KMeans(n_clusters = self.n_categories)
            catval = km.fit_predict(X)

        self.clfMaster_ = LogisticRegression(penalty='l2',C=1)
        self.clfMaster_.fit(X,catval)
        catpred = self.clfMaster_.predict(X)
        
        self.clfCat_ = []
        for i in list(set(catval)):
            clfIndividual = LogisticRegression(penalty='l2',C=1)
            ind = np.where(catpred == i)[0]
            clfIndividual.fit(X[ind,:],y.iloc[ind])
            self.clfCat_.append(clfIndividual)
       
        return self

    def predict(self, X):
        catval = self.clfMaster_.predict(X)
        pred = np.empty(X.shape[0],dtype='object')
        for i in list(set(catval)):
            ind = np.where(catval == i)[0]
            p = self.clfCat_[i].predict(X[ind,:])
            pred[ind] = p
        return pred

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,threshold=0.4):
        self.threshold = threshold
        pass

    def fit(self, X, y):
        self.v_ = CountVectorizer(token_pattern=r"[^,]+",ngram_range=(1,2))
        self.v2_ = CountVectorizer()
        train_vec = hstack((self.v_.fit_transform(X),self.v2_.fit_transform(X)))
        self.clfLR_ = LogisticRegression(penalty='l2',C=1)
        self.clfLR_.fit(train_vec,y)
        self.clfUN_ = UniqueItemClassifier()
        self.clfUN_.fit(X,y)

    def predict(self, X):
        pred = np.empty(X.shape[0],dtype='object')
        
        predUN, scores = self.clfUN_.predict(X,probability=True)
        pred_vec = hstack((self.v_.transform(X),self.v2_.transform(X)))
        predLR = self.clfLR_.predict(pred_vec)
        maxScore = list(max(x.values()) for x in scores)

        ind = (np.array(maxScore) >= self.threshold)
        pred[ind] = predUN[ind]
        pred[~ind] = predLR[~ind]

        return pred



def evalPred(val_pred,val):
    cmp = np.core.defchararray.equal(val_pred.astype(str),val['cuisine'].values.astype(str))
    return np.sum(cmp.astype(int))/float(len(cmp))


# Create training vectors from the training data
v = CountVectorizer(token_pattern=r"[^,]+")
v_bigram = CountVectorizer(token_pattern=r"[^,]+",ngram_range=(1,2))
v_words = CountVectorizer()
train = pd.read_json('train.json')
train['ingredients'] = train['ingredients'].apply(lambda x: ','.join(x))
train_vec = v.fit_transform(train['ingredients'])
train_vec_bigram = v_bigram.fit_transform(train['ingredients'])
train_vec_words = v_words.fit_transform(train['ingredients'])
train_vec_combined = hstack((train_vec_bigram,train_vec_words))



clfLR = LogisticRegression(penalty='l2',C=1,multi_class='multinomial',solver='lbfgs');

print "Testing logistic regression"
print cross_val_score(clfLR,train_vec_combined,train['cuisine'])

print "Testing Nested LR"
clfNLR = NestedLogisticRegressionClassifier()
print cross_val_score(clfNLR,train_vec_combined,train['cuisine'])

print "Testing Unique Item Classifier"
clfUN = UniqueItemClassifier()
print cross_val_score(clfUN,train['ingredients'],train['cuisine'])

print "Testing Ensemble Classifier"
clfEn = EnsembleClassifier(threshold=0.75)
print cross_val_score(clfEn,train['ingredients'],train['cuisine'])

cat1 = ['chinese','japanese','thai','vietnamese','filipino','korean']
cat2 = ['mexican','indian','moroccan','jamaican','brazilian','spanish']
cat3 = ['italian','french','greek']
cat4 = ['southern_us','cajun_creole','british','irish','russian']

x_train, x_val, y_train, y_val = train_test_split(train_vec_combined,train['cuisine'],test_size=0.3)

print "training logistic regression"
clf.fit(train_vec_combined,train['cuisine'])

test = pd.read_json('test.json')
test['ingredients'] = test['ingredients'].apply(lambda x: ','.join(x))
test_vec_bigram = v_bigram.transform(test['ingredients'])
test_vec_words = v_words.transform(test['ingredients'])
test_vec_combined = hstack((test_vec_bigram,test_vec_words))

print "predicting"
pred = clfLR.predict(test_vec_combined)

out = pd.DataFrame()
out['id'] = test['id']
out['cuisine'] = pred
out.to_csv("submission_2.csv",index=False)
