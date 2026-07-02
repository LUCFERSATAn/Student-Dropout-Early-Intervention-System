Student Dropout Early Intervention System
This project is an early intervention system designed to identify students at risk of dropping out based on their academic performance and financial status. By predicting student risk early, educators can provide timely support and counseling.

Installation
Clone this repository to your local machine.

Install the required dependencies using the requirements.txt file:

Bash
pip install -r requirements.txt
Usage
To run the Streamlit application, execute the following command in your terminal:

Bash
streamlit run app.py
Features
Risk Prediction: Uses a machine learning model to estimate the risk of student dropout.

Interactive Dashboard: Built with Streamlit to allow easy input of student data and visualization of risk factors.

Intervention Recommendations: Provides actionable recommendations based on the calculated risk level.

Project Structure
app.py: The main Streamlit application script.

dataset.csv: The dataset used for model training and analysis.

dropout_model.pkl: The pre-trained machine learning model file.

requirements.txt: The list of Python dependencies required to run the project.
