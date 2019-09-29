# Disaster Response Pipelines Project

Motivation : This project was completed as part of the Udacity Data Science NanoDegree

Background: The data is sourced from Figure8
The dataset contains 30,000 messages from disaster events in multiple languages, encoded with 36 different categories. 

more detail here : https://www.figure-eight.com/dataset/combined-disaster-response-data/

Outcome: The aim of this project was to build a simple web application that could take a message and assign categories in order to suggest which forms of aid are required. 

<h3> Installation : </h3>

git clone https://github.com/julsie01/DisasterResponsePipelines.git

<h4> Libraries Used </h4>

The project is built with python 3.6 using the following libraries
SQl Alchemy, Scikit-learn, XGBoost, sqllite, pandas, nltk, regex, flask, plotly

If installing manually in a conda environment then the following commands can be used

conda install numpy <br>
conda install sqlalchemy <br>
conda install -c anaconda scikit-learn <br>
conda install -c anaconda sqlite <br>
conda install pandas <br>
conda install nltk <br>
conda install regex <br>
conda install flask <br>
conda install plotly <br>
conda install py-xgboost <br>

To Install from requirements file: <br>

cd DisasterResponsePipelines <br>

If using pip <br>

sudo python3 -m pip install -r requirements/requirements.txt <br>
install text package for use with the machine learning script <br>
pip install -e text_process

If using conda <br>

conda install pip <br>
conda env create --name disasterresponseenv -f=requirements/requirements.txt -v <br>
conda activate disasterresponseenv <br>

install text package for use with the machine learning script <br>
pip install -e text_process

download nltk packages <br>
python -m nltk.downloader stopwords <br>
python -m nltk.downloader wordnet <br>
python -m nltk.downloader punkt <br>


<H4> Files Included: </H4>

<H5>Data Cleaning</H5>

data_process folder contains a script to load data from csv files and transfer into an sql lite database.

<H5>Text Processing</H5>

text_process folder contains a package to tokenie, stem and lemmatise text.

<H5>Model Training and Evaluation</H5>

model folder contains scripts to run a text processing pipeline (using the text_process package) and train a multi output classifer

<H5>Web App</H5>

app folder contains a small flask web application that allows a user to input a message and it is classified into the relevant categories by the trained model.  

#The following instructions assume that you are in the DisasterResponsePipelines Directory

<h5> To Run the data cleaning script on linux use the following command </h5>
python data_process/process_data.py ./data/messages.csv ./data/categories.csv ./data/DisasterResponse.db

<H5> To Run the model training use the following command <br> </h5>
Running a grid search is option as the grid search is long running if you are only using a cpu <br>
If you want to run the grid search then change the 0 to 1 <br>

python model/train_classifier.py data/DisasterResponse.db models/classifier.pkl 0

<br>
<h5>To Run the flask web app: <br></h5>
move to the app directory <br>
     cd app     <br>
Run the following command. <br>
    python run.py <br>

Go to http://0.0.0.0:3001/ <br>
