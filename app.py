from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

app = Flask(__name__)

df = pd.read_csv("./divine.properties.alt.csv")

df = df[["price","bedrooms","bathrooms","sqft_living","sqft_lot","city","statezip","sqft_basement","sqft_above","view","floors","condition","waterfront","yr_built","yr_renovated"]]

class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        return X
    
X = df.drop(["price"], axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

label_encoder = ['city', 'statezip']
scaler = [col for col in X.columns if col not in label_encoder]

preprocessor = ColumnTransformer(
    transformers=[
        ('label_encode', MultiColumnLabelEncoder(columns=label_encoder), label_encoder),
        ('scale', StandardScaler(), scaler)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

pipeline.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        bedrooms = request.form['bedrooms']
        bathrooms = request.form['bathrooms']
        sqft_living = request.form['sqft_living']
        sqft_lot = request.form['sqft_lot']
        sqft_basement = request.form['sqft_basement']
        sqft_above = request.form['sqft_above']
        city = request.form['city']
        statezip = request.form['statezip']
        view = request.form['view']
        floors = request.form['floors']
        condition = request.form['condition']
        waterfront = request.form['waterfront']
        yr_built = request.form['yr_built']
        yr_renovated = request.form['yr_renovated']

        input_data = pd.DataFrame({
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'sqft_living': [sqft_living],
            'sqft_lot': [sqft_lot],
            'sqft_basement': [sqft_basement],
            'sqft_above': [sqft_above],
            'city': [city],
            'statezip': [statezip],
            'view': [view],
            'floors': [floors],
            'condition': [condition],
            'waterfront': [waterfront],
            'yr_built': [yr_built],
            'yr_renovated': [yr_renovated],
        })

        prediction = pipeline.predict(input_data)[0]

        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
