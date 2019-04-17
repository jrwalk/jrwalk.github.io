---
title: A Simple, Dockerized ML Model Serving API
layout: post_norel
permalink: /pages/projects/model-serving
---

Rather than directly interfacing ML models with a broader production code
environment, it is often simpler to present model inference through a REST
API -- this confers a number of benefits, including isolating the runtime
environment of the model from the broader production
environment (ensuring version/language independence) and greatly simplified
model versioning & retraining.  Wrapping the API in a
[Docker](https://www.docker.com/) container also simplifies deployment &
development by providing a self-contained testing, development, and runtime
environment.

My intention in building this [simple model-serving API](https://github.com/jrwalk/model-serving)
using Flask was an exercise in [test-driven development](https://en.wikipedia.org/wiki/Test-driven_development)
practices, as well as a place to tinker with `docker-compose` and other
production-grade container management practices.

## the model

For simplicity, the API uses scikit-learn [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
objects combined with the [`sklearn-pandas`](https://github.com/scikit-learn-contrib/sklearn-pandas)
package's `DataFrameMapper` to systematize the model interface.  The pipeline
must begin with (at least one) `DataFrameMapper` object - this encodes any
preprocessing transforms needed, and allows feature selection by name from the
input JSON.  Conversely, the last step of the pipeline must be an
`sklearn.BaseEstimator` instance for the final prediction (any steps in between
need only support the `sklearn` transformer API).

## Docker usage

The service is built on top of the `python3.7-slim` image using `docker-compose`, which provides three services:

1. `box` - base build service and general-purpose debugging interface to
the container.
2. `test` - Runs the `pytest` suite of unit tests for the service.
3. `app` - starts the Flask debug webserver.

Simply running `docker-compose up` from the root directory will trigger all three services.

To load a model into new container, simply ensure that the model is packed via
`cloudpickle` into a file called `pipeline.pkl` in the `./binary` directory - this is mounted as a volume to the container and is immediately accessible.

## to-dos & next steps

1. additional "production-grade" service (gunicorn or similar)
2. transition to using [FastAPI](https://fastapi.tiangolo.com/) for
aynchronous serving (includes `uvicorn` webserver)
3. more unit testing!
4. better support for model upload/control
