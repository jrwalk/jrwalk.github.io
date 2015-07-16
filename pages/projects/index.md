---
title: Projects
layout: default
---

# Projects
{% for post in site.posts %}
  - {{ post.date | date: '%B %Y' }} <span class="separator">~</span> [{{ post.title }}]({{ post.url }})
{% endfor %}
