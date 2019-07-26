FROM python:3
ADD application.py /
ADD requirements.txt /
RUN pip install --user --upgrade setuptools
RUN pip install -r requirements.txt
EXPOSE 8080
CMD [ "python", "./application.py" ]
