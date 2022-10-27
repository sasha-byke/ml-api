# Machine Learning prediction API cloud solution in a docker container with preprocessing module in a secure python package as an Azure artifact

This blog is to give an idea on how to solve the following problem: what if you need a machine learning prediction API solution in a docker container but in order to build said image you need a private python package (for example, preprocessing, or any other module that you would need in a more than 1 repo\solution and that contains sensitive information) that should be securely installed using Azure pipelines and Azure artifacts. The idea is as follows: we create a python package, put it in an Azure artifact feed then we either use Personal Access Tokens from azure to install it locally or use Azure pipelines to authenticate for us.


## Introduction

The following is to show how to create:
1) Simple XGBoost machine learning model, store it in a pickle file
2) Your own python package with preprocessing module, store it in Azure artifacts in a private repository
3) Create an API in python using FastAPI with the pickled model
4) Build a docker container with the API, securely install your private package using Azure pipelines

Why would you need a separate package for preprocessing? At HumanTotalCare we work with medical data, thus the security of our software and our packages are essential -- and a separate python package is needed so that we could use the same preprocessing model for different solutions (for example, training and prediction services). The same logic would apply to any reusable module that you would want to use as a python package.


## 1) Machine Learning Model
Here, we will use the publicly available model and dataset from here https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/ to train and store the model in a pickle file. 
The code is suitable for Python version 3.9

Now we will dive into folder `api`. Train on the data from "data" folder and save output as a pickle in a "model" folder

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

# split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# fit model on training data
model = XGBClassifier()
model.fit(X_train, y_train)

# skipping testing part as it is out of scope for this blog.
    
filename = "models/model_diabetes.pkl"

# save
pickle.dump(model, open(filename, "wb"))
```

Run the code
```bash
$ python train.py
```

## 2) Private python package in Azure for preprocessing
Before we start, it's good to read the latest guide on how to upload a python package on microsoft's website: 
https://docs.microsoft.com/en-us/azure/devops/pipelines/tasks/package/twine-authenticate?view=azure-devops

First, let's look at the files structure and give explanations for every important piece here: 

```bash
└───preprocessing
    │   build-pipeline.yml
    │   pyproject.toml
    │   README.md
    │   setup.cfg
    │
    └───src
        └───mypreprocessing
                preprocessing.py
                __init__.py
```

We start with a simple preprocessing function that just converts a python list to a NumPy array

`src/mypreprocessing/preprocessing.py`:
```python
import numpy as np

def preprocess(data):
    
    # Convert data to the format you need, the following is a gimmick 
    # here, you can apply scaler, do one-hot encoding and other things
    x = np.array(data)

    return x
```

`pyproject.toml` is quite simple, just do not forget to include packages that you need (like numpy, scikit-learn or others)

Here we specify the name of our package (so `pip` knows how to call it), some credentials and the location of our modules (where = `src`) 
`setup.cfg`:
```conf
[metadata]
name = mypreprocessing
version = 1.1.0
author = HumanTotalCare B.V. , R&D department, Data Science team
author_email = a.bykov@humantotalcare.nl
description = Public example repo for preprocessing

[options]
package_dir =
    = src
packages = find:
python_requires = >=3.9

[options.packages.find]
where = src
```

This is a good moment to pause for a bit to make some preparations -- make sure you have your artifact feed created, for that use the following guide from Microsoft:
https://docs.microsoft.com/en-us/azure/devops/artifacts/quickstarts/python-packages?view=azure-devops#create-a-feed

In this example, we created `dss` (data science service) feed and will use it explicitly later on

And last but not least, we specify a build pipeline for Azure pipelines:
`build-pipeline.yaml`:
```yaml
trigger:
  - "main"

pool:
  vmImage: ubuntu-latest

steps:
- task: UsePythonVersion@0
  inputs:
    versionSpec: '3.9'
  displayName: 'Use Python 3.9'

- script: |
    python -m pip install --upgrade pip
    python -m pip install --upgrade build setuptools twine
    python -m build
  displayName: 'Install dependencies and build'


- task: TwineAuthenticate@1
  inputs:
    artifactFeed: dss

- script: |
   twine upload -r "dss" --config-file $(PYPIRC_PATH) dist/*
  displayName: 'Publish artifact'
```
There are 3 important things here: 
- `python -m build` builds the package for us and puts all the files into `dist` folder, 
- `TwineAuthenticate@1` does the authentication 
- `twine upload` stores our package as an artifact in Azure

Now we have our package securely stored in Azure but the problem is we can't just `pip install` it by design in our container, we need to either supply a secret as a personal access token (PAT) to install it locally in a docker image or use azure pipelines authentication to do so. We will discuss that in detail later

## 3) Create an API
First, let's decide what our API input (request) and output (response) are. It is a good practice to provide your API consumer with an explicit data schema -- here we do so by using recommended BaseModel as FastAPI suggests. We explicitly define every feature that we expect by name and data type. For simplicity, in a response class, we define only the model name and prediction value (1 or 0)

Refer to FastAPI documentation (https://fastapi.tiangolo.com/) for more information on the subject.

From here we go back to folder `api`, where we will compile our `main.py` from several pieces of code:

First, define our data schema:
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

class Response(BaseModel):
    model: str
    pred: int
```

Next step is to load model once from a pickle file and put it in a variable 
```python
import pickle

filename="models/model_diabetes.pkl"
model = pickle.load(open(filename, "rb"))
```

And now the actual FastAPI implementation -- we create just one POST method that takes data in `Request` format and outputs data in `Response` format as has been previously defined in data schema.

Here we create an array from `Request` but we convert it to a NumPy array via our private preprocessing package `mypreprocessing`
```python
from fastapi import FastAPI
import numpy as np

from mypreprocessing import preprocess_data

app = FastAPI()

@app.post("/predict", response_model=Response)
def predict(item: Request):

    data = [
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

    x = preprocess_data(data)
    
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

# import function from 
from mypreprocessing import preprocess_data

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

    data = [
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

    x = preprocess_data(data)


    
    prediction = model.predict(x)
    
    prediction_response = {
                "model": "diabetes",
                "pred": prediction[0]
            }
    
    return prediction_response
```

## 4) Dockerfile, build an image, install private package:
### Requirements
First, lets put all the python libraries (including our private one) that we used in `requirements.txt` :
```txt 
fastapi
numpy
scikit-learn
xgboost
uvicorn
pydantic
typing
pandas

mypreprocessing==1.1.0
```

## Dockerfile
Create Dockerfile with our API. Here, it takes `python:3.9` image, adds a current folder to it and makes it a working directory, installs python libraries from requirements.txt and finally starts our API using the same command as we would use to test our API locally

There is nothing specific about Dockerfile we will use, except the fact that we need to install our private package in the image. Here we will cover 2 options for doing that: first, local, is to supply Personal Access Token in a secret file -- that would allow us to build the image locally. Even though it is not a preferred method, we still wanted to cover that to understand "how" it is done on the cloud as well which is hidden from our eyes by Azure. The second option (preferred) is to use Azure tools for that when you use build agents. We will cover both options in detail below
### Option 1: Local

#### Prepare local secret from PAT (Personal Access Token)
(Used this is a reference https://medium.com/devops-dudes/using-private-python-azure-artifacts-feeds-in-alpine-docker-builds-909e6558c1a4)

1. Create Personal Access Token in azure, copy the secret ("pattoken") locally
Instructions: https://docs.microsoft.com/en-us/azure/devops/organizations/accounts/use-personal-access-tokens-to-authenticate?view=azure-devops&tabs=Windows#create-a-pat

2.  Create a `pip.url.secret` file in the root folder that contains the following
```pip.url.secret
https://<feed_name>:<pattoken>@pkgs.dev.azure.com/<organisation>/_packaging/<feed_name>/pypi/simple/
```
that token will be manually added to your docker image

Be aware that this is a temporary measure and running this in production is very unsecure -- the token is very powerful. There are also ways to remove it from the container afterwards, but that is a little too deep for this blog to cover

`Dockerfile_local`:
```Dockerfile
FROM python:3.9

ADD . /api
WORKDIR /api

# be aware this is not very secure and should be used very accurately
RUN PRIVATE_PIP_AZURE_ARTIFACTS_URL=$(cat pip.url.secret) && \
    pip install -r requirements.txt --extra-index-url=${PRIVATE_PIP_AZURE_ARTIFACTS_URL}  

# start the server
CMD ["uvicorn", "main:app", "--port", "5000", "--host", "0.0.0.0"]
```
So let's break down what happens here, first of all, parameter `--extra-index-url` for `pip` is the way to say to pip to not download packages from the "default" repository but to actually go to a private one (for us, that would be our Azure artifacts). In order to communicate to that storage, we create an environmental variable `PRIVATE_PIP_AZURE_ARTIFACTS_URL` that is basically storing the PAT (secret) and a domain name of our artifact. We supply inside of our `pip.url.secret` to that variable so that our docker image knows where to look for our preprocessing package and how to authenticate itself to install the package

#### Build image
```bash
$ docker build -f Dockerfile -t ml-api:latest .
```

#### Run container
Here we use once again port 5000 for consistency
```bash
$ docker container run -p 5000:5000 ml-api:latest
```
#### Test
Test with the same command as before:
```bash
$ curl -X POST -H "Content-Type: application/json" --url http://localhost:5000/predict -d @mock.json

{"model":"diabetes","pred":1}
```

#### Swagger UI page
From both container and local API versions, you have your swagger UI version exposed for other API consumers to know how to communicate with your API. By default it is available here (in your browser):

https://localhost:5000/docs


#### Run the API locally:
Do not forget to install your preprocessing first!
```bash
$ uvicorn main:app --port 5000 --host 0.0.0.0
```

#### Test your API
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

### Option 2: Cloud with Azure pipelines (preferred)
This option is relatively easier to set up as you do not need to touch any secrets explicitly

#### Dockerfile
So the Dockerfile would look much simpler:
`Dockerfile`:
```Dockerfile
FROM python:3.9

ADD . /api
WORKDIR /api

RUN pip install -r requirements.txt

# start the server
CMD ["uvicorn", "main:app", "--port", "5000", "--host", "0.0.0.0"]
```
#### Build pipeline
`build-pipeline.yml`:
```yaml
trigger:
- '*'

pool:
  vmImage: ubuntu-latest

variables:
  imageName: ml-api:latest


steps:
- task: UsePythonVersion@0
  inputs:
    versionSpec: '3.9'
  displayName: 'Use Python 3.9'

- task: PipAuthenticate@1
  displayName: Authenticate with artifact feed
  inputs:
   artifactFeeds: dss
   onlyAddExtraIndex: true

- bash: |
    docker build \
      --build-arg 'INDEX_URL= $(PIP_EXTRA_INDEX_URL)' \
      -t $(imageName) \
      -f Dockerfile \
      .
  displayName: Build image
```
Once again, let us break a few things down here. First of all, task `PipAuthenticate@1` is doing the authentication for us, we specify artifact feed (`dss`) and it produces an environmental variable for us `PIP_EXTRA_INDEX_URL` that contains an address and access token to connect to private Azure artifact storage (similar to one that we put in `pip.url.secret` locally before)

Next, that variable is supplied to `docker build` command with argument `--build-arg 'INDEX_URL= $(PIP_EXTRA_INDEX_URL)'` so that `pip` inside the docker image knows where to look for our package and how to authenticate itself in order to install it

As soon as you have your image built, you can push it to your container storage. 

## Conclusion
We learnt how to create and test an API running a machine learning model, securely store preprocessing modules for it in a private python package in Azure as an artifact and how to create a containerised docker solution with all of those locally or in the cloud.
