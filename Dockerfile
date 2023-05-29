FROM python:3.11.3-bullseye

WORKDIR /app

RUN apt update -qq
RUN apt-get update
RUN apt install r-cran-rstan --assume-yes

RUN R -e "install.packages('generics')"
RUN R -e "install.packages('stats')"
RUN R -e "install.packages('forecast')"
RUN R -e "install.packages('ggplot2')"
RUN R -e "install.packages('bayesplot')"
RUN R -e "install.packages('bayesforecast')"

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]