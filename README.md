
# Disaster Response Pipeline 

This project is part of Udacity Data Science Nano Degree. The goal of this project is to classify the message into the multiple-output category and create web app where people can enter the message and get the result predicted category of the message. 


### Installization 
* sklearn 
* nltk
* sqlalchemy 
* pandas, 


### Instructions:
Run the following commands in the project's root directory to set up your database and model.

	* 	To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
	* 	To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
	*	 Run the following command in the app's directory to run your web app. python run.py
  *	Go to http://0.0.0.0:3001/

### Project Files:

	* data/process_data.py: The ETL pipeline used to process data in preparation for model building.
	*	models/train_classifier.py: The Machine Learning pipeline used to fit, tune, evaluate, and export the model to a Python pickle (pickle is not uploaded to the repo due to size constraints.).
	*	app/templates/*.html: HTML templates for the web app.
	*	run.py: Start the Python server for the web app and prepare visualizations.
