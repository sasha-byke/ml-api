# TL; DR:
## Option 1: run API locally with uvicorn
```bash
$ uvicorn main:app --port 5000 --host 0.0.0.0
```
## Option 2 run API with model in a docker container locally:
Build image
```bash
$ docker build -f Dockerfile -t ml-api:latest .
```
Run container 
```bash
$ docker container run -p 5000:5000 ml-api:latest
```
## Test with json file with data (works for both options, change values in mock.json if needed)
```bash
$ curl -X POST -H "Content-Type: application/json" --url http://localhost:5000/predict -d @mock.json

{"model":"diabetes","pred":1}
```
# In-depth explanations
Here we used publicly available model and dataset from here https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/ to make a simple machine learning API solution out of model using python FastAPI in a docker container. Repo is tried and tested for Python version 3.9

## Train model using XGBoost
Train on the data from "data" folder and save output as a pickle in a "model" folder
```bash
$ python train.py
```

Code for `train.py`:
```python
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# load data
df = pd.read_csv('data/pima-indians-diabetes.csv')

# split data into X and y
# X contains all columns but the last one, last column is our label y
X, y = df.iloc[:, :-1], df.iloc[:, [-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

filename = "models/model_diabetes.pkl"

# save
pickle.dump(model, open(filename, "wb"))
```

## Design the API
First, let's decide what our API input (request) and output (response) are. It is a good practice to provide your API consumer with an explicit data schema -- here we do so using recommended BaseModel as FastAPI suggests. We explicitly define every feature that we expect by name and data type. For simplicity, in a response class we define only the model name and prediction value (1 or 0)

Refer to FastAPI documentation for more information on the subject.

```python
from pydantic import BaseModel

class Request(BaseModel):
    feature_a: int
    feature_b: int
    feature_c: int
    feature_d: int
    feature_e: int
    feature_f: float
    feature_g: float
    feature_h: int

class Prediction(BaseModel):
    model: str
    pred: int
```

Next step is to load model once from a pickle file and put it in a variable 
```python
import pickle

filename="models/model_diabetes.pkl"
model = pickle.load(open(filename, "rb"))
```

And now the actual Fast API implementation -- we create just one POST method that takes data in `Request` format and outputs data in `Response` format.

Here we create a numpy array from `Request` item (that is just an example, you can do your own data (pre-)processing as needed.

As soon as the array `x` is assembled, we feed it to our model's predict method and supply the result to the prediction response

```python
from fastapi import FastAPI
import numpy as np

app = FastAPI()

@app.post("/predict", response_model=Response)
def predict(item: Request):

    x = np.array(
        [
            [
                item.feature_a, 
                item.feature_b, 
                item.feature_c, 
                item.feature_f, 
                item.feature_e, 
                item.feature_f, 
                item.feature_g, 
                item.feature_h
            ]
        ]
    )
    
    prediction = model.predict(x)
    
    prediction_response = {
                "model": "diabetes",
                "pred": prediction[0]
            }
    
    return prediction_response

```

Complete code for `main.py`:
```python
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

class Request(BaseModel):
    feature_a: int
    feature_b: int
    feature_c: int
    feature_d: int
    feature_e: int
    feature_f: float
    feature_g: float
    feature_h: int

class Response(BaseModel):
    model: str
    pred: int


filename = "models/model_diabetes.pkl"
model = pickle.load(open(filename, "rb"))

app = FastAPI()

@app.post("/predict", response_model=Response)
def predict(item: Request):

    x = np.array(
        [
            [
                item.feature_a, 
                item.feature_b, 
                item.feature_c, 
                item.feature_f, 
                item.feature_e, 
                item.feature_f, 
                item.feature_g, 
                item.feature_h
            ]
        ]
    )
    
    prediction = model.predict(x)
    
    prediction_response = {
                "model": "diabetes",
                "pred": prediction[0]
            }
    
    return prediction_response
```

## Run the API locally:
```bash
$ uvicorn main:app --port 5000 --host 0.0.0.0
```

## Test your API
You can use example `mock.json` file for testing:
```json
{
    "feature_a":"0",
    "feature_b":"121",
    "feature_c":"66",
    "feature_d":"30",
    "feature_e":"165",
    "feature_f":"34.3",
    "feature_g":"0.203",
    "feature_h":"33"
}
```
And in a separate terminal test it via cURL:
```bash
$ curl -X POST -H "Content-Type: application/json" --url http://localhost:5000/predict -d @mock.json

{"model":"diabetes","pred":1}
```

## Create Dockerfile, build an image, run the container:

### Requirements
First, lets put all the python libraries that we used in `requirements.txt` :
```txt 
fastapi
numpy
scikit-learn
xgboost
uvicorn
pydantic
typing
pandas
```

### Dockerfile
Second, create Dockerfile with our API. Here, it takes `python:3.9` image, adds current folder to it and makes it a working directory, installs python libraries from requirements.txt and finally starts our API using the same command as we used to test our API locally

`Dockerfile`:
```Dockerfile
FROM python:3.9

ADD . /ml-api
WORKDIR /ml-api

RUN pip install -r requirements.txt

# start the server
CMD ["uvicorn", "main:app", "--port", "5000", "--host", "0.0.0.0"]
```

### Build image
```bash
$ docker build -f Dockerfile -t ml-api:latest .
```
### Run container
Here we use once again use port 5000 for consistency
```bash
$ docker container run -p 5000:5000 ml-api:latest
```
### Test
Test with the same command as before:
```bash
$ curl -X POST -H "Content-Type: application/json" --url http://localhost:5000/predict -d @mock.json

{"model":"diabetes","pred":1}
```

## Swagger UI page
From both container and local API versions, you have your swagger UI version exposed for other API consumers to know how to communicate with your API. By default it is available here (in your browser):

https://localhost:5000/docs


