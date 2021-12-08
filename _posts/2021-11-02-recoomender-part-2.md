---
title: End-To-End Recommender Systems, Part II
layout: post_norel
permalink: /pages/projects/recommender-part-2
description: "Building Recommender systems, part II."
---

In [part I](/pages/projects/recommender-part-1), we discussed building a collaborative-filtering recommender model based on the open-source MovieLens dataset.
Ultimately, we reduced the problem to simple computation based on embedding vectors for our users and movies, such that recommendations could be inferred agnostic of the actual embedding process.
These embeddings are represented by a few model artifacts:

- the embeddings themselves (written to our database)
- an `annoy` approximate-nearest-neighbors index of the movie embeddings for quick lookup
- mappings of user/movie IDs to ordinal indices corresponding to the embedding vectors

In this post, we'll examine how we can use these embeddings to reliably serve recommendations in a production setting that can scale to large traffic volumes.
I've [already written](/pages/projects/flask-to-fastapi) about using [FastAPI](https://fastapi.tiangolo.com/) for building model-serving APIs, so we'll use that as a starting point.
To make our recommender model production-ready, however, we'll need to add a bit more complexity to the system -- in this post, we'll walk through building our architecture out of multiple containerized services for a performant, scalable serving API.

Example code for this serving architecture can be found [here](https://github.com/jrwalk/recommender-demo).
Note that we won't exhaustively cover our code in the blog -- I'll just call out relevant or interesting components, but the example code in the repo will function as-is.
Let's dive in!

## A Note on Containerization

While good containerization practices, e.g. with a tool like [Docker](https://www.docker.com/), would fill several posts in their own right, we should take a moment to consider it for this project.
I've [written a bit on this](/pages/projects/managing-python-environments) before -- [containerization](https://www.docker.com/resources/what-container) is an incredibly useful concept for managing dependencies down to the system level, which goes a long way towards getting out of the vagaries of managing Python (or any other) environment.
Pretty much any modern cloud deployment scheme supports (or requires) containerized services, so by developing locally in containers we can easily ship our individual components off to the appropriate cloud service, while [docker-compose](https://docs.docker.com/compose/) lets them interact in a realistic way for local development.

We'll build up our architecture incrementally, but it's worth getting a high-level view of the containerized services in our example code:

- `adhocs`: scripts for environment setup, e.g. database migrations, data bootstrapping, and model builds. In production, this would be handled by deploy pipelines and controlled ETL/task frameworks like [Apache Airflow](https://airflow.apache.org/).
- `postgres`: a containerized PostgreSQL database, which we'll use as our at-rest store for both raw data (i.e., interactions) and processed data like user and movie embeddings. Uses a standard container image.
- `redis`: a [Redis](https://redis.io/) in-memory key/value store, which we'll use for fast caching. Uses a standard container image.
- `app`: the Python container running our FastAPI app for model serving.
- `app-celery`: a separate Python container running [Celery](https://docs.celeryproject.org/en/stable/getting-started/introduction.html) background tasks for analytics.

### Python Docker Containers

A number of our services (like our datastores, PostgreSQL and Redis) provide stock Docker images that we can use as-is for our services.
Realistically we won't deploy these services directly -- rather, in production we'd use managed cloud services like [Aurora](https://aws.amazon.com/rds/aurora/) and [ElastiCache](https://aws.amazon.com/elasticache/) on AWS, or [Cloud SQL](https://cloud.google.com/sql) and [Memorystore](https://cloud.google.com/memorystore) in GCP, so we really only need local stand-ins for development purposes.

For our main app, though, we need to build our container from scratch, so it's worth taking some time to make sure we get it right.
We want to hit three main points:

- First, as we've [discussed previously](/pages/projects/managing-python-environments), the Alpine images commonly found in production usage for other languages are fraught with problems, so we'll instead use the debian-based slim Python images.
- Next, we'll use [Poetry](https://python-poetry.org/) for our dependency resolution.
At present, `pip` still struggles with resolving installed package versions, and can result in broken environments if you're not careful -- obviously, we want to avoid this in production.
Poetry both presents more intelligent dependency resolution, and a single unified [PEP-518](https://www.python.org/dev/peps/pep-0518/) compliant specification file in `pyproject.toml`.
- Finally, we can treat Poetry properly as a build dependency, as well as filter out some of our install requirements, by using a [multi-stage build](https://pythonspeed.com/articles/smaller-python-docker-images/) -- basically, we'll create within a single Dockerfile the spec both for a build image that installs all our dependencies, and a deployment image that grabs only what it needs at runtime from the build image.

In the example Dockerfile below, we grab (and clean up) our build dependencies, as well as a pinned version of Poetry, then create a virtual environment for Poetry to install dependencies into.
Note that we don't immediately activate the virtual environment -- each Docker command is run as a separate process, so simply calling the usual `source /opt/venv/bin/activate` won't persist through the rest of the build process.
Instead, we manually force the virtual environment to the front of the search path, so it will effectively be the global default Python for the rest of the container commands (we do, however, have to inline activate it along with `poetry install` for Poetry to behave properly).
For our deploy image, we can then just grab the virtual environment from `/opt/venv` and pull it wholesale into the new image -- we've abstracted away the usage of Poetry for dependency resolution, and can treat the created environment as the normal default Python env in the container from here forward.

```docker
###############
# build image #
###############

FROM python:3.9.5-slim-buster as build-image

WORKDIR /opt/app

RUN \
    apt-get update \
    && apt-get install -y gcc g++ curl libpq-dev \
    && rm -rf /var/lib/apt/lists/*

ENV \
    POETRY_VERSION=1.1.11 \
    POETRY_HOME=/opt/poetry

RUN \
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

ENV PATH=$POETRY_HOME/bin:$PATH

COPY ./pyproject.toml /opt/app/pyproject.toml
COPY ./poetry.lock /opt/app/poetry.lock

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN . /opt/venv/bin/activate && poetry install --no-root

################
# deploy image #
################

FROM python:3.9.5-slim-buster as deploy-image

WORKDIR /opt/app

RUN \
    apt-get update \
    && apt-get install -y libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build-image /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY . /opt/app
```

In our `docker-compose.yml` specification, we target specific images from the Dockerfile like so:

```yaml
services:
  app-base: &app-base
    build:
      context: ./app
      target: deploy-image
```

By removing extraneous build requirements from our final image, we can both achieve significantly faster builds (provided the `build-image` is appropriately cached) and smaller deployed artifacts -- the final `deploy-image` here is just over 400MB, where an equivalent single-stage build would be well over 1GB.

## The Serving API

With our container squared away, we're ready to start building out app.
Like we've [seen before](/pages/projects/flask-to-fastapi), setting up a FastAPI app is straightforward: we instantiate a `FastAPI` object and bind route functions to it to define the API endpoints.
We'll start this in `app.py`:

```python
# app.py

from fastapi import FastAPI

from .schemas import (
    Healthcheck,
    RecommendationRequest,
    RecommendationResponse,
)

app = FastAPI(
    title="serving-app",
    version="0.1.0",
    description="a simple serving app for a recommender model",
)


@app.get("/health-check", response_model=Healthcheck)
async def healthcheck() -> Healthcheck:
    """A simple high-state healthcheck."""
    return Healthcheck(status="ok")


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(
    request_body: RecommendationRequest,
) -> RecommendationResponse:
    """Retrieves the requested number of recommendations for a given user.

    If no data can be found for a user, returns an empty set of recommendations.
    """
    raise HTTPException(status_code=500, detail="we haven't built this yet!")
```

Where we've created the app, a "healthcheck" endpoint that just indicates the app is alive, and the skeleton of our recommendation endpoint.
Defining an API route is simply a matter of defining a function for the endpoint's operation and binding it to the app via the decorator -- for example, the `recommend` function will present itself via an HTTP request like `curl -X POST $API_URL/recommend`.

```python
# schemas.py

from uuid import UUID

from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass


# request schemas
class RecommendationRequest(BaseModel):
    user_id: int
    cutoff: int = 10


# response schemas
@dataclass
class Healthcheck:
    status: str


@dataclass
class Recommendation:
    movie_id: int
    title: str
    genres: list[str]
    score: float = Field(..., ge=0, le=1)


@dataclass
class RecommendationResponse:
    request_id: UUID
    recommendations: list[Recommendation]
```

## Recommender Model Inference

<p align="center">
  <img src="/images/projects/recommender/flow.svg" />
</p>

## Testing
