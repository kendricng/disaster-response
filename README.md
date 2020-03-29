# An NLP Classification Task to Disaster Response

# Table of Contents

1. [Installation](https://github.com/kendricng/disaster-response#1-installation)
2. [Repo Motivation](https://github.com/kendricng/disaster-response#2-repo-motivation)
3. [File Descriptions](https://github.com/kendricng/disaster-response#3-file-descriptions)
4. [Summary of Results](https://github.com/kendricng/disaster-response#4-summary-of-results)
5. [Others](https://github.com/kendricng/disaster-response#5-others)

# 1. Installation

The scripts are written in Python 3.7+ using the below libraries:

## Python Libraries

- [numpy](https://numpy.org/): mathematical computations and array manipulation; and
- [pandas](https://pandas.pydata.org/): data manipulation and analysis
- [re] : manipulation of regular expressions as strings
- [nltk](https://www.nltk.org/): natural language toolkit for processing text;
- [pickle](https://docs.python.org/3/library/pickle.html): Python object serialization tool used to pickle machine learning results; and
- [scikit-learn](https://scikit-learn.org/): machine learning training and pipeling building.
- [flask](https://flask.palletsprojects.com/en/1.1.x/): micro web framework;
- [json](https://docs.python.org/2/library/json.html): manipulation of JSON formats; and
- [plotly](https://plot.ly/): scientific graphic libraries.
- [SQLAlchemy](https://www.sqlalchemy.org/): database access and writing; and
- [sys](https://docs.python.org/2/library/sys.html): scripts' interaction with the Python interpreter.

Note that dependency issues have been recorded while running `scikit-learn` from version `0.19.1` onwards.

# App Instructions

1. Run the following commands in the project's root directory:

- To run ETL pipeline that cleans data and stores in database, run `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- To run ML pipeline that trains classifier and saves a pickle file, run `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
- To run the web app, run `python app/run.py`

2. Go to http://0.0.0.0:3001/

# 2. Repo Motivation

This repository predicts the type of disaster a particular message is related to using Natural Language Processing techniques. It was motivated by the Udacity Data Science Nanodegree and its partnership with [Figure Eight](https://www.figure-eight.com/).

# 3. File Descriptions

This repository consists of the following files:

- [app](./app): front end files to render onto a website;
- [data](./data): original CSV files from Figure Eight, the database file from the combined CSV files, and the script used to generate the database; and
- [model](./model): script for training the machine learning model.

Note that the pickle file was too large to upload onto GitHub. We suggest you training the model on local with a stronger GPU capacity as it may take hours to train the model.

# 4. Summary of Results

## Technical Summary

I performed NLP on the messages and used TF-IDF and Count to use as features for predicting the label associated with that message (e.g. first-aid, water, natural disaster).

Without GridSerach, the model had an average F-1 score of around 0.8. After adding GridSearch with 2-3 hyperparameters, we improved the F-1 score to around 0.89.

## Feature Engineering

I improved the score of this model mainly by using the following classifiers:

1. Count
2. TFIDF
3. Multiclass Classification

## Room for Improvement

- Faster run times with `SpaCy` and `TensorFlow` instead of using a `TF-IDF` Vectorizer.
- Diagnose new version issues with `scikit-learn`.
- Over- and undersampling messages from certain labels to have a more balanced dataset.

# 5. Others

## License

[MIT License](LICENSE)

## Acknowledgements

I would like to acknowledge the following parties who have inspired and motivated me to create this repository:

- [Udacity Data Science Nanodegree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025): for providing the program structure to push me to create this repository; and
- Other contributors and peers (anonymous) who provided feedback on the technical and non-technical pieces of the project.

## Discussion and Feedback

For any questions, comments, and feedback on this repository, both technical and non-technical, feel free to reach out to me at Twitter [@KendricNg](https://twitter.com/KendricNg).
