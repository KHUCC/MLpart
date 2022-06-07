FROM python:3.9

ADD . /www
WORKDIR /www

RUN python -m pip install --upgrade pip 
RUN pip3 install Flask
RUN pip3 install uwsgi
RUN pip3 install boto3
RUN pip3 install flask_restful
RUN pip3 install tinify


CMD uwsgi uwsgi.ini