---
title: Projects
layout: default
subtitle: John Walk
---

![ProfilePhoto](/images/headers/rainier.jpg){: .inline-img}

# Projects
{% for post in site.posts %}
  - {{ post.date | date: '%B %Y' }} <span class="separator">~</span> [{{ post.title }}]({{ post.url }})
{% endfor %}
