FROM python:3.7

LABEL maintainer="paloma.piot@udc.es"

RUN apt update && apt-get install -y python3 

COPY ./requirements.txt /profiler/requirements.txt

WORKDIR /profiler

RUN pip install -r requirements.txt 

COPY . /profiler

EXPOSE 5000

ENTRYPOINT [ "python" ]

CMD [ "profiler.py" ]