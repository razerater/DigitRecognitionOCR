# Optical Character Recognition and Facial Recognition
Final project for Intro to AI (CSCI 4150) at RPI. Spring 2019.

Run the Perceptron Learning Algorithm on the dataset of handwritten digits:
```
python dataClassifier.py -c perceptron --autotune
```
Run the Naive Bayes classifier with Laplace smoothing (additive smoothing) on the same dataset:
```
python dataClassifier.py -c naiveBayes --autotune
```
Run the MIRA classifier on the same dataset:
```
python dataClassifier.py -c mira --autotune
```
To run any of the above classifiers on the faces dataset, just add `-d faces`.

Note: this project uses Python 2.7.
