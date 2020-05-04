# Purpose

The world knowledge QA example shows you how to create a question answering system using the Wikidata world knowledge graph as information source.

# Files

`neuralmodels`: folder containing code for the neural networks which perform prediction of relation, topic entities and relation direction
`domain.py`: file containing information about the ontology and code for looking up information in the Wikidata world knowledge graph
`multinlg.py`: file containing code for natural language generation; differs from default NLG by allowing multiple system utterances
`policyqa.py`: file containing code for deciding which system acts to perform; differs from default Policy by not using a BST and allowing multiple system acts
`semanticparser.py`: file containing code for natrual language understanding; uses the neural models for predicting relation, topic entities and relation direction and creating user acts accordingly
