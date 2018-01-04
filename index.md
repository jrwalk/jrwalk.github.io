---
layout: default
title: John Walk
---

![ProfilePhoto](/images/headers/faroes.jpg){: .inline-img}

<p>Engineer, scientist, developer, tinkerer, adventurer.</p>

<p>I'm interested in clean energy technology, open-source code development, 
data science, exploring the frontiers of plasma physics, aerospace, and 
computer building.  I'm currently a researcher in 
<a href="/pages/fusionprimer">nuclear fusion energy</a> at 
MIT's Alcator C-Mod tokamak experiment, where I have worked in varying 
capacities since my sophomore year, completing my doctorate in September
2014.  Working in fusion means new challenges in extreme environments on some of the most advanced devices ever built - I'm always interested in new projects and problems to solve.</p>

<p>When I'm not coding or in the lab, I like to spend my time hiking, rock 
climbing, or mountaineering.  Having the White Mountains in NH so close is one 
of the best parts of working at MIT!</p>

## Projects

{% for post in site.posts limit:5 %}
  - {{ post.date | date: '%B %Y' }} <span class="separator">~</span> [{{ post.title }}]({{ post.url }})
{% endfor %}
[more projects...](/pages/projects)

## Skills

**Programming Languages:** Python, SQL, Bash, Go, LaTeX, some Scala, 
Java/Groovy, Javascript

**Data Processing & Pipelines:** Pandas, Dask, Luigi, Apache Spark

**Machine Learning & Statistics:** scikit-learn, XGBoost, H2O, Tensorflow/Keras, 
PyMC3, Pomegranate 

**Natural Language Processing:** NLTK, TextBlob, Gensim, CoreNLP

**Visualization & Dashboards:** Matplotlib, Seaborn, Bokeh, D3.js, 
Metabase, Superset 

**Hosting & Deployment:** Flask, Docker, AWS Sagemaker

**Experimental Physics:** data analysis, numerical modeling, laser optics, 
X-ray and IR spectroscopy, Thomson Scattering spectrometer systems

**Teaching:** laboratory supervision, review/recitation classroom sessions for 
introductory circuit design for MIT Nuclear Science & Engineering undergraduates
