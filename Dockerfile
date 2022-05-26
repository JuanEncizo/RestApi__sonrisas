FROM python:3.10.3

RUN python -m pip install --upgrade pip
    

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD [ "python", "src/app.py" ]

