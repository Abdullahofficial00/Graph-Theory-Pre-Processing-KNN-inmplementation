from sklearn.model_selection import train_test_split
from gspan_mining.config import parser
from gspan_mining.main import main
import networkx as nx
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
import functions


# Combine all your documents into a single list
allDocuments = functions.getDocuments()
labels = 15 * ['travel'] + 15 * ['desease'] + 15 * ['hobby']


nx.draw(allDocuments[0], with_labels=True)
# splitting testing and training set
X_train, X_test, y_train, y_test = train_test_split(allDocuments, labels, test_size=0.2, random_state=33)

# extracting particular graphs
businessGraph = nx.compose_all([doc for doc, label in zip(X_train, y_train) if label == 'travel'])
foodGraph = nx.compose_all([doc for doc, label in zip(X_train, y_train) if label == 'hobby'])
healthGraph = nx.compose_all([doc for doc, label in zip(X_train, y_train) if label == 'desease'])
# write graphs in file for gSpan
nx.write_edgelist(businessGraph, 'travelGraph.txt')
nx.write_edgelist(foodGraph, 'hobbyGraph.txt')
nx.write_edgelist(healthGraph, 'deseaseGraph.txt')
# gspan argument 
travelArgs = '-s 2 -d True -l 5000 -p True -w True travelGraph.txt'
hobbyArgs = '-s 2 -d True -l 5000 -p True -w True hobbyGraph.txt' 
deseaseArgs = '-s 2 -d True -l 5000 -p True -w True deseaseGraph.txt'
# Use gSpan to extract frequent subgraphs from each category graph
travelSubgraphs,_ = parser.parse_known_args(args=travelArgs.split())
hobbySubgraphs,_ = parser.parse_known_args(args=hobbyArgs.split())
deseaseSubgraphs,_ = parser.parse_known_args(args=deseaseArgs.split())
travelSubgraphs = main(travelSubgraphs)
hobbySubgraphs = main(hobbySubgraphs)
deseaseSubgraphs = main(deseaseSubgraphs)
# # converting the subgraphs to binary vectors
travelSubgraphs = [functions.graphToVector(subgraph) for subgraph in travelSubgraphs._frequent_subgraphs]
hobbySubgraphs = [functions.graphToVector(subgraph) for subgraph in hobbySubgraphs.graphs]
deseaseSubgraphs = [functions.graphToVector(subgraph) for subgraph in deseaseSubgraphs.graphs]
# print(travelSubgraphs)
# print(hobbySubgraphs)
# print(deseaseSubgraphs)
# For each document in the training set, calculate the Jaccard similarity between the document graph and the frequent subgraphs of each category
try:
  for docGraph, trueLabel in zip(X_train, y_train):
    travelSimilarity = jaccard_score(docGraph, travelSubgraphs, average='weighted')
    hobbySimilarity = jaccard_score(docGraph, hobbySubgraphs, average='weighted')
    deseaseSimilarity = jaccard_score(docGraph, deseaseSubgraphs, average='weighted')
    # print(travelSimilarity, hobbySimilarity, deseaseSimilarity)
    # Assigning the document to the category with the highest similarity
    if max(travelSimilarity, hobbySimilarity, deseaseSimilarity) == travelSimilarity:
      predictedLabel = 'travel'
    elif max(travelSimilarity, hobbySimilarity, deseaseSimilarity) == hobbySimilarity:
      predictedLabel = 'hobby'
    else:
      predictedLabel = 'desease'
    # Compare the predicted label with the true label to evaluate the performance of the model
    if predictedLabel == trueLabel:
      predictedLabel = 'Correct prediction'
    else:
      predictedLabel='Incorrect prediction'

  # Initialize counters for correct and total predictions
  correct_predictions = 0
  total_predictions = 0

  # Repeat the process for the test set
  for docGraph, trueLabel in zip(X_test, y_test):
    travelSimilarity = jaccard_score(docGraph, travelSubgraphs, average='weighted')
    hobbySimilarity = jaccard_score(docGraph, hobbySubgraphs, average='weighted')
    deseaseSimilarity = jaccard_score(docGraph, deseaseSubgraphs, average='weighted')
    
    # Assign the document to the category with the highest similarity
    if max(travelSimilarity, hobbySimilarity, deseaseSimilarity) == travelSimilarity:
      predictedLabel = 'travel'
    elif max(travelSimilarity, hobbySimilarity, deseaseSimilarity) == hobbySimilarity:
      predictedLabel = 'hobby'
    else:
      predictedLabel = 'desease'
    
    # Compare the predicted label with the true label
    if predictedLabel == trueLabel:
      correct_predictions += 1
    total_predictions += 1

  # Calculate the accuracy of the model on the test set
  accuracy = correct_predictions / total_predictions
  print(f"Accuracy on test set: {accuracy * 100}%")
except:
  # Here, 80% of the data will be used for training and 20% will be used for testing
  X_train, X_test, y_train, y_test = train_test_split(allDocuments, labels, test_size=0.2, random_state=33)

  

  # Extract features from each graph
  X_train_features = np.array([functions.extractGraphFeatures(graph) for graph in X_train])
  X_test_features = np.array([functions.extractGraphFeatures(graph) for graph in X_test])
  predictions = functions.calculateAccuracy(X_train_features,y_train,X_test_features,y_test)
  # Compute confusion matrix
  confMatrix = confusion_matrix(y_test, predictions)
  functions.plotConfusionMatrix(confMatrix)

