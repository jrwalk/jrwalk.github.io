---
layout: default
title: John Walk
---

<p>Physicist and Mountain Climber!</p>

##Projects
{% for post in site.posts limit:5 %}
  - {{ post.date | date: '%B %Y' }} <span class="separator">~</span> [{{ post.title }}]({{ post.url }})
{% endfor %}
[more projects...](/pages/projects)

##Current Research Topics
list topics here (bullets?)

##Skills
bullet of skills