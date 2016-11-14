import time
import datetime
import numpy as np
import scipy as sp
import pandas as pd
import csv
import sklearn as sl
import sklearn.feature_extraction.text 
import sklearn.svm
import sklearn.cross_validation
import sys
import notifier
import nltk
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import random
from nltk.corpus import reuters


now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

outputDir = "outSvm"
fileOut = "./"+outputDir+"/metricsSvm-"+timestamp+".txt"
fileOutDetail = "./"+outputDir+"/metricsSvmDetails-"+timestamp+".txt"
fileTokens = "./"+outputDir+"/tokensSvm-"+timestamp+".txt"
filePrecRecall = "./"+outputDir+"/precisionRecallSvm-"+timestamp+".pdf"


cutOff = 3
tokenPattern = '(?u)\\b\\w*[a-zA-z_][a-zA-Z_]+\\w*\\b'

random_state = np.random.RandomState(0)

stemmer = nltk.stem.snowball.EnglishStemmer(ignore_stopwords=True)
analyzer = sl.feature_extraction.text.CountVectorizer(analyzer='word', token_pattern=tokenPattern).build_analyzer()
#modAnalyzer = lambda doc: (stemmer.stem(w) for w in analyzer(doc))
modAnalyzer = analyzer

time0 = time.time()

print("{0:.1f} sec - Creating dataframe".format(time.time()-time0))
dfReuters = pd.DataFrame(columns=['text','category'])

i=0
for cat in reuters.categories():
        for doc in reuters.fileids(cat):
                dfReuters.loc[i] = [reuters.raw(doc), cat]
                i = i+1

print("{0:.1f} sec - Tokenizing".format(time.time()-time0))
countVec = sklearn.feature_extraction.text.CountVectorizer(min_df=cutOff, strip_accents='unicode', analyzer=modAnalyzer, stop_words=stemmer.stopwords)
termMatrix = countVec.fit_transform(dfReuters.text)

#dfTerm = pd.DataFrame(termMatrix.toarray(), columns=countvec.get_feature_names())

#sys.exit()

np.savetxt(fileTokens, countVec.get_feature_names(), '%s')

print("{0:.1f} sec - Training".format(time.time()-time0))
targetUnb = dfReuters.category.tolist()
target = sl.preprocessing.label_binarize(targetUnb, np.unique(targetUnb))
#target = np.array(list(map(str, dfReuters.category.tolist())))

trainMatrix, testMatrix, trainTarget, testTarget = sl.cross_validation.train_test_split(termMatrix.tocsr(), target, test_size=.1, random_state=random_state)

#clf = sklearn.multiclass.OneVsRestClassifier(sklearn.svm.SVC(kernel='linear', probability=True, random_state=random_state))
clf = sklearn.multiclass.OneVsRestClassifier(sklearn.svm.LinearSVC(random_state=random_state))

yScore = clf.fit(trainMatrix, trainTarget).decision_function(testMatrix)

precision, recall, threshold = sklearn.metrics.precision_recall_curve(testTarget.ravel(),yScore.ravel())
#precision, recall, threshold = sklearn.metrics.precision_recall_curve(testTarget,yScore[:0])
average_precision = sklearn.metrics.average_precision_score(testTarget, yScore, average="micro")

plt.clf()
ax = plt.figure().gca()
ax.set_xticks(np.arange(0,1,0.05))
ax.set_yticks(np.arange(0,1.,0.05))
plt.xticks(rotation='vertical')
plt.plot([0.,1.],[0.,1.], color='blue')
plt.plot(recall, precision, color='gold', lw=2, label='micro-average Precision-recall curve (area = {0:0.2f})'''.format(average_precision))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower right")
plt.grid()
plt.savefig(filePrecRecall)

notifier.sendMessage("Terminated "+timestamp+" execution ({0:.1f} sec)".format(time.time()-time0))

print("{0:.1f} sec - End".format(time.time()-time0))

sys.exit()
#print("{0:.1f} - Score".format(time.time()-time0))
#scoreSede = clfSede.score(testMatrix, testTargetSede)
#scoreMorfologia = clfMorfo.score(testMatrix, testTargetMorfo)

print("{0:.1f} sec - Predict".format(time.time()-time0))
testPredictSede = clfSede.predict(testMatrix)
testPredictMorfo = clfMorfo.predict(testMatrix)

print("{0:.1f} sec - Metrics".format(time.time()-time0))
out_file = open(fileOut, 'w')

out_file.write("f1 score sede micro average: ")
out_file.write(str(sl.metrics.f1_score(list(map(str, testTargetSede)), list(map(str, testPredictSede)), average='micro')))
out_file.write("\nf1 score sede macro average: ")
out_file.write(str(sl.metrics.f1_score(list(map(str, testTargetSede)), list(map(str, testPredictSede)), average='macro')))

out_file.write("\nf1 score morfologia micro average: ")
out_file.write(str(sl.metrics.f1_score(list(map(str, testTargetMorfo)), list(map(str, testPredictMorfo)), average='micro')))
out_file.write("\nf1 score morfologia macro average: ")
out_file.write(str(sl.metrics.f1_score(list(map(str, testTargetMorfo)), list(map(str, testPredictMorfo)), average='macro')))
out_file.write("\n")
out_file.close()

out_file_details = open(fileOutDetail, 'w')

out_file_details.write(sl.metrics.classification_report(list(map(str, testTargetSede)), list(map(str, testPredictSede))))
out_file_details.write("\n")
out_file_details.write(sl.metrics.classification_report(list(map(str, testTargetMorfo)), list(map(str, testPredictMorfo))))

out_file_details.close()

notifier.sendFile("Terminated "+timestamp+" execution ({0:.1f} sec)".format(time.time()-time0), fileOut)

print("{0:.1f} sec - End".format(time.time()-time0))
