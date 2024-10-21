# set base image (host OS)
FROM python:3.9

# copy the content of the local src directory to the working directory
COPY ./ .

# install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install protobuf==3.20.0
RUN pip install -r docReq.txt

RUN apt update -y
RUN apt install -y vim

RUN python manage.py migrate 

# command to run on container start
CMD ["python", "manage.py", "runserver", "127.0.0.1:8080"]
