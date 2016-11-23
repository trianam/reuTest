import time
import datetime
import numpy as np
import sklearn as sl
import sklearn.feature_extraction.text 
import sklearn.svm
import sklearn.model_selection
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


cutOff = 2
tokenPattern = '(?u)\\b\\w*[a-zA-z_][a-zA-Z_]+\\w*\\b'

random_state = np.random.RandomState(42)

stemmer = nltk.stem.snowball.EnglishStemmer(ignore_stopwords=True)
analyzer = sl.feature_extraction.text.CountVectorizer(analyzer='word', token_pattern=tokenPattern).build_analyzer()
modAnalyzer = lambda doc: (stemmer.stem(w) for w in analyzer(doc))

time0 = time.time()

print("{0:.1f} sec - Creating dataset".format(time.time()-time0))

catIndex = dict()

for i,cat in enumerate(reuters.categories()):
    catIndex[cat] = i    

X = np.empty(len(reuters.fileids()), dtype=object)
Y = np.zeros((len(reuters.fileids()), len(reuters.categories())))

for i,fid in enumerate(reuters.fileids()):
    X[i] = reuters.raw(fid)
    for cat in reuters.categories(fid):
        Y[i,catIndex[cat]] = 1

trainX, testX, trainY, testY = sl.model_selection.train_test_split(X, Y, test_size=.2, random_state=random_state)

print("{0:.1f} sec - Tokenizing".format(time.time()-time0))
vectorizer = sl.feature_extraction.text.TfidfVectorizer(min_df=cutOff, max_df=0.5, max_features=9947, analyzer=modAnalyzer, stop_words=stemmer.stopwords)
trainXV = vectorizer.fit_transform(trainX)
testXV = vectorizer.transform(testX)

np.savetxt(fileTokens, vectorizer.get_feature_names(), '%s')

print("{0:.1f} sec - Training".format(time.time()-time0))
clf = sl.multiclass.OneVsRestClassifier(sl.svm.LinearSVC(random_state=random_state))

yScore = clf.fit(trainXV, trainY).decision_function(testXV)


precision, recall, threshold = sl.metrics.precision_recall_curve(testY.ravel(),yScore.ravel())
average_precision = sl.metrics.average_precision_score(testY, yScore, average="micro")

indexClass = dict()
precisionClass = dict()
recallClass = dict()
average_precisionClass = dict()

classesToPlot = ['earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn']

for className in classesToPlot:
    index = catIndex[className]

    precisionClass[className], recallClass[className], _ = sl.metrics.precision_recall_curve(testY[:,index],yScore[:,index])
    average_precisionClass[className] = sl.metrics.average_precision_score(testY[:,index], yScore[:,index])

numClasses = testY.shape[1]
precisionAll = dict()
recallAll = dict()
average_precisionAll = dict()
for i in range(numClasses):
    precisionAll[i], recallAll[i], _ = sl.metrics.precision_recall_curve(testY[:,i],yScore[:,i])
    average_precisionAll[i] = sl.metrics.average_precision_score(testY[:,i], yScore[:,i])

plt.clf()
ax = plt.figure().gca()
ax.set_xticks(np.arange(0,1,0.05))
ax.set_yticks(np.arange(0,1.,0.05))
plt.xticks(rotation='vertical')
plt.plot([0.,1.],[0.,1.], color='blue')
for i in range(numClasses):
    plt.plot(recallAll[i], precisionAll[i], color='black')
for className in classesToPlot:
    plt.plot(recallClass[className], precisionClass[className], lw=2, label='"{0}",{1:0.2f}'''.format(className, average_precisionClass[className]))

plt.plot(recall, precision, color='gold', lw=2, label='microAvg,{0:0.2f}'''.format(average_precision))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
ax = plt.axes()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

plt.legend(loc="upper left", bbox_to_anchor=(1., 1.), fancybox=True, shadow=True)
plt.grid()
plt.savefig(filePrecRecall)

print("{0:.1f} sec - Predict".format(time.time()-time0))
testPredict = clf.predict(testXV)

print("{0:.1f} sec - Metrics".format(time.time()-time0))
out_file = open(fileOut, 'w')

out_file.write("precision score micro average: ")
out_file.write(str(sl.metrics.precision_score(testY, testPredict, average='micro')))
out_file.write("\nrecall score micro average: ")
out_file.write(str(sl.metrics.recall_score(testY, testPredict, average='micro')))
out_file.write("\nf1 score micro average: ")
out_file.write(str(sl.metrics.f1_score(testY, testPredict, average='micro')))
out_file.write("\nprecision score macro average: ")
out_file.write(str(sl.metrics.precision_score(testY, testPredict, average='macro')))
out_file.write("\nrecall score macro average: ")
out_file.write(str(sl.metrics.recall_score(testY, testPredict, average='macro')))
out_file.write("\nf1 score macro average: ")
out_file.write(str(sl.metrics.f1_score(testY, testPredict, average='macro')))
out_file.write("\n")
out_file.close()

out_file_details = open(fileOutDetail, 'w')
out_file_details.write(sl.metrics.classification_report(testY, testPredict))
out_file_details.write("\n")
out_file_details.close()

notifier.sendFile("Terminated "+timestamp+" execution ({0:.1f} sec)".format(time.time()-time0), fileOut)

print("{0:.1f} sec - End".format(time.time()-time0))

