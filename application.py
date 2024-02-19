from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Defining the Flask application
application = Flask(__name__)

app=application

# Route to handle the home page
@app.route('/')
def index():
    """
    Main method to render the home page.
    """
    return render_template('index.html')

@app.route('/home', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Method to predict the data point.
    """
    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        # Get the data from the form
        data=CustomData(
            
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
            
        )
        
        # Get the data as dataframe
        pred_df = data.get_data_as_df()
        print(pred_df) 
        
        # Predict the data point and return the results
        predict_pipeline = PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])

if __name__ == '__main__':
    app.run(host="0.0.0.0")
        