FROM python:3.9

ADD . /dss-public-api
WORKDIR /dss-public-api

RUN pip install -r requirements.txt

# start the server
CMD ["uvicorn", "main:app", "--port", "5000", "--host", "0.0.0.0"]
