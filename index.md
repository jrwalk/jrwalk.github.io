---
layout: default
title: About
subtitle: John Walk
---

![ProfilePhoto](/images/headers/faroes.jpg){: .inline-img}

## Hello.

I am a senior data scientist currently working at 
[Cinch Financial](https://cinchfinancial.com/), building a one-of-a-kind personal 
financial advisor to level the playing field with banks and credit card issuers. 
As the first hire on the DS team (joined April 2016), I have had a hand in every 
aspect of Cinch's development as a data-driven company, including designing 
end-to-end pipelines for machine-learning solutions from raw data 
ingestion/processing through modeling to production implementation, designing 
analytics dashboards & data visualizations, and even creating custom training 
materials for data analytics tools (Python, SQL, etc.) for junior business/data 
analysts.  I'm always looking for new ways to expand my toolkit & experience for 
ML & deep learning, statistical/Bayesian modeling, data visualization, and 
scalable data pipelines.

In my previous life, I was a researcher in 
[nuclear fusion energy](/pages/fusionprimer) at the MIT Plasma Science & Fusion 
Center's [Alcator C-Mod tokamak experiment](http://www.psfc.mit.edu/research/magnetic-fusion-energy), 
where I worked in various capacities since my Sophomore year, completing my 
doctorate in September 2014.  Following academic postdoctoral research at MIT,
I made the conversion to data science through the 
[Insight Data Science](http://insightdatascience.com/) program, an intensive 
postdoctoral fellowship geared towards STEM PhDs transitioning to data-driven 
industry, with the Boston session I attended particularly geared towards 
healthcare & biomedical applications -- I was particularly drawn to the 
opportunity for data science in socially-beneficial applications.

Outside of work, I spend my time hiking, rock & ice climbing, and 
mountaineering.  While most of my experience is in the excellent White Mountains 
of New Hampshire, I've also logged ascents of Mt. Rainier (14,411') in Washington, 
and Ixtaccíhuatl (17,159') and Pico de Orizaba (18,491') in Mexico.


## Projects

{% for post in site.posts limit:5 %}
  - {{ post.date | date: '%B %Y' }} <span class="separator">~</span> [{{ post.title }}]({{ post.url }})
{% endfor %}
[more projects...](/pages/projects)

## Skills & Tools

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
