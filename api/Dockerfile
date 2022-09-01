FROM python:3.9

RUN pip install -r requirements.txt

ADD . /api
WORKDIR /api

# start the server
CMD ["uvicorn", "main:app", "--port", "5000", "--host", "0.0.0.0"]
