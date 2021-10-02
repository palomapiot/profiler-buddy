# Profiler Buddy

## About

Profiler Buddy is a service for automatic profiling of social media users from the analysis of unstructured data. It allows you to predict the demographic data from the user's feed and it automatically fills Beck Depression Inventory, based on the user's post history.

## Features

Input data: list of user social posts (natural language texts)

- Demographic data profiling
  - Gender with score
- Beck Depression Inventory answer prediction
  - Predict BDI answers
  - Score and context each answer hit

## API

Check the [OpenAPI specification](https://github.com/palomapiot/profiler-buddy/blob/develop/openapi.yaml) for more information regarding the API.

## Prerequisites

- Python 3
- pip
- Install dependencies with `pip install -r requirements.txt`

or run as a container with:

- Docker
- docker-compose

## Installation

Clone the project and start the application with `python profiler.py` or run as a container with `docker-compose up`

## Roadmap

- Include more demographic data:
  - Age
  - Location
  - Personality
  - Social status
  - Wealth

- Improve prediction accuracies:
  - n-grams and n-chars approach
  - bag of words

- Assist for other mental disorders:
  - Anorexia
  - Suicidal toughts
  - Anxiety

## Useful links

- Google developer group: https://groups.google.com/u/1/g/early-dev
- Beck Depression Inventory: https://www.ismanet.org/doctoryourspirit/pdfs/Beck-Depression-Inventory-BDI.pdf
- Gender classification paper: https://www.scitepress.org/Link.aspx?doi=10.5220/0010431901030113

## License

GNU GPLv3.0
