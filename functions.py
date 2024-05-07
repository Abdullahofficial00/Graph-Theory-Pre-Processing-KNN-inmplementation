import networkx as nx
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import  TfidfVectorizer
# from mcs import getMcs
# globar variable
def assessClassificationPerformance(predictedLabels,testLabels):
  accuracy = accuracy_score(testLabels, predictedLabels)
  precision = precision_score(testLabels, predictedLabels, average='weighted')
  recall = recall_score(testLabels, predictedLabels, average='weighted')
  f1 = f1_score(testLabels, predictedLabels, average='weighted')
  return {
      "accuracy": accuracy,
      "precision": precision,
      "recall": recall,
      "f1Score": f1
  }

def prepareTextData(doc):
  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform(doc)
  return X, vectorizer

def classifyText(xTrain, yTrain, xTest):
  nb = MultinomialNB()
  nb.fit(xTrain, yTrain)
  yPred = nb.predict(xTest)
  return yPred


# functions
def getDocuments():
    count=0
    deseaseGraphsList=[]
    travelGraphsList=[]
    hobbyGraphsList=[]
    while count<3:
        for i in range(15):
            if(count==0):
                fileName='ProcessedTravelData/processedTravelDoc'+str(i)+'.txt'
            elif(count==1):
                fileName='ProcessedHobbyData/processedHobbyDoc'+str(i)+'.txt'
            elif(count==2):
                fileName='ProcessedDeseaseData/processedDeseaseDoc'+str(i)+'.txt'
            doc=''
            with open(fileName,'r',encoding='utf-8') as f:
                doc=f.read()
                doc=doc.split()
            G = nx.Graph()
            G.add_nodes_from(doc)
            for i in range(len(doc) - 1):
                word1 = doc[i]
                word2 = doc[i + 1]
                G.add_edge(word1, word2)
            if(count==0 ):
                travelGraphsList.append(G)
            elif(count==1):
                hobbyGraphsList.append(G)
            elif(count==2):
                deseaseGraphsList.append(G)
        count=count+1
    return travelGraphsList+hobbyGraphsList+deseaseGraphsList
def getProcessedDocs():
    count=0
    deseaseGraphsList=[]
    hobbyGraphsList=[]
    travelGraphsList=[]
    while count<3:
        for i in range(15):
            if(count==0):
                fileName='ProcessedTravelData/processedTravelDoc'+str(i)+'.txt'
            elif(count==1):
                fileName='ProcessedHobbyData/processedHobbyDoc'+str(i)+'.txt'
            elif(count==2):
                fileName='ProcessedDeseaseData/processedDeseaseDoc'+str(i)+'.txt'
            doc=''
            with open(fileName,'r',encoding='utf-8') as f:
                doc=f.read()
            if(count==0 ):
                travelGraphsList.append(doc)
            elif(count==1):
                hobbyGraphsList.append(doc)
            elif(count==2):
                deseaseGraphsList.append(doc)
        count=count+1
    return travelGraphsList,hobbyGraphsList,deseaseGraphsList

def plotGraph(graphs):
    for G in graphs:
        nx.draw(G, with_labels=True)   
        plt.show()
# Convert the graphs to binary vectors
def graphToVector(graph):
    return [1 if graph.has_edge(u, v) else 0 for u, v in nx.complete_graph(graph.nodes).edges]
def vectorToGraph(vector):
    G = nx.Graph()
    for edge in vector:
        G.add_edge(*edge)
    return G
def extractGraphFeatures(graph):
      numNodes = graph.number_of_nodes()
      numEdges = graph.number_of_edges()
      return np.array([numNodes, numEdges])

def calculateAccuracy(xTrainFeatures,yTrain,xTestFeatures,yTest):

  knnClassifier = KNeighborsClassifier(n_neighbors=3)

  # Train the classifier
  knnClassifier.fit(xTrainFeatures, yTrain)

  # Predict on the test set
  predictions = knnClassifier.predict(xTestFeatures)

  # Calculate accuracy
  accuracy = accuracy_score(yTest, predictions)
  print("Accuracy: {:.4f}".format(accuracy*100),"%")
  return predictions

# to update
def getScore( g1, g2):
        matchingGraph = nx.Graph()
        for n1, n2 in g2.edges():
            if g1.has_edge(n1, n2):
                matchingGraph.add_edge(n1, n2)
        components = list(nx.connected_components(matchingGraph))
        return sum([len(i) for i in components]) / min(g1.number_of_nodes(), g2.number_of_nodes())

def predictUsingKnn(trainGraphs, testGraphs):
  scores = []
  for category, graphs in trainGraphs.items():
    print('category',category)
    print('graphs',graphs)
    for graph in graphs:
      score = getScore(testGraphs, graph)
      scores.append((category, score))
  # Sort by decreasing score (highest score first)
  scores.sort(key=lambda x: x[1], reverse=True)
  # Extract top K categories
  topCategories = [category for category, _ in scores[:30]]

  # Find most frequent category
  majorityLabel = Counter(topCategories).most_common(1)[0][0]

  return majorityLabel

def performKnnClassification( trainGraphs, testGraphs):
  yPred = []
  yTest = []
  for category, graphs in testGraphs.items():
    for graph in graphs:
      prediction = predictUsingKnn(trainGraphs, graph)
      yPred.append(prediction)
      yTest.append(category)

  return yPred, yTest

def plotConfusionMatrix(confMatrix):
  plt.figure(figsize=(8, 6))
  sns.heatmap(confMatrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Travel', 'Hobby', 'Desease'], yticklabels=['Travel', 'Hobby', 'Desease'])
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix')
  plt.show()



