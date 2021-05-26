---
layout: default
title: About
subtitle: John Walk
---

![ProfilePhoto](/images/headers/faroes.jpg){: .inline-img}

## Hello.

#### who am I?

I'm a data scientist & ML engineer.

#### what do I love?

Building scalable data solutions, solving natural-language processing and search problems, and teaching Python and software engineering best practices to other data scientists.

#### what's my story?

I'm currently a senior machine learning engineer in the Boston office for [Pluralsight](https://www.pluralsight.com/) supporting the Discovery team, powering content search, browse, and exploration experiences to help our learners explore and unlock new tech skills.
Previously, I was a manager / technical lead at Wayfair on the Merchandising Ops Data Science team, providing data-driven solutions for managing, maintaining, and capitalizing on structured & unstructured information about millions of Wayfair.com products.

Prior to Wayfair, my first Data Science role was as a data scientist at Cinch Financial, a small startup building a one-of-a-kind personal financial advisor to level the playing field with banks and credit card issuers.
I joined Cinch through the [Insight Data Science](http://insightdatascience.com/) program, an intensive postdoctoral fellowship geared towards STEM PhDs transitioning to data-driven industry.
The Boston session I attended was geared towards healthcare & biomedical applications -- I was particularly drawn to the opportunity for data science in socially-beneficial applications.

My academic background is in nuclear engineering & plasma physics, and before transitioning to data science I was a researcher in [nuclear fusion energy](/pages/fusionprimer) at the MIT Plasma Science & Fusion Center's [Alcator C-Mod tokamak experiment](http://www.psfc.mit.edu/research/magnetic-fusion-energy), where I had worked in various capacities since my Sophomore year, completing my doctorate in September 2014.

Outside of work, I spend much my time hiking, rock & ice climbing, and mountaineering.
While most of my experience is in the excellent White Mountains of New Hampshire, I've also logged ascents of Mt. Rainier (14,411') in Washington, and Ixtaccíhuatl (17,159') and Pico de Orizaba (18,491') in Mexico.
I also enjoy board and tabletop games, and have worked for several years as an Enforcer at PAX East and Unplugged teaching new & unreleased board games.

## Projects

{% for post in site.posts limit:5 %}
  - {{ post.date | date: '%B %Y' }} <span class="separator">~</span> [{{ post.title }}]({{ post.url }})
{% endfor %}

[more projects...](/pages/projects)

## Skills & Tools

**Programming Languages:**
Python, Rust, Cython, C++, Bash, Javascript (Node, React)

**Neural Networks & Deep Learning:**
Tensorflow, Keras, PyTorch

**Machine Learning & Statistics:**
scikit-learn, XGBoost, H2O, PyMC3, Pomegranate, LightFM

**Natural Language Processing:**
FastText, SpaCy, NLTK, Gensim, CoreNLP

**Data Technologies**
Elasticsearch, SQL (Postgres, MS SQL), Hive, Vertica, Apache Kafka, AWS Athena

**Data Processing & Pipelines:**
Apache Airflow, Dask, Apache Spark, Pandas, Luigi

**Visualization & Dashboards:**
Altair, Matplotlib, Seaborn, Bokeh, D3.js, Metabase, Superset

**Hosting & Deployment:**
FastAPI, Flask, Docker, Kubernetes, AWS, Gitlab CI/CD, TeamCity

**Experimental Control**
LaunchDarkly, Unleash
