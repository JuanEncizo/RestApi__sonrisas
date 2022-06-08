FROM python:3.10.3

RUN python -m pip install --upgrade pip
    

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install libgl1 -y

CMD [ "python", "application.py" ]

