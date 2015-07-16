---
layout: default
title: John Walk
subtitle: MIT PSFC
---

<p>Engineer, scientist, developer, tinkerer, adventurer.</p>

<p>I'm interested in clean energy technology, open-source code development, 
data science, exploring the frontiers of plasma physics, aerospace, and 
computer building.  I'm currently a researcher in nuclear fusion energy at 
MIT's Alcator C-Mod tokamak experiment, where I have worked in varying 
capacities since my sophomore year, completing my doctorate in September 2014.  
I'm always interested in new challenges!</p>

<p>When I'm not coding or in the lab, I like to spend my time hiking, rock 
climbing, or mountaineering.  Having the White Mountains in NH so close is one 
of the best parts of working at MIT!</p>

##Projects
{% for post in site.posts limit:5 %}
  - {{ post.date | date: '%B %Y' }} <span class="separator">~</span> [{{ post.title }}]({{ post.url }})
{% endfor %}
[more projects...](/pages/projects)

##Current Research Topics
* Experimental work generating high-resolution measurements of the *pedestal* 
in high-performance plasmas for fusion power plants
* Computational modeling of the large-scale stability and turbulence in the 
plasma edge
* developing new open-source, cross-machine data analysis tools for fusion 
researchers

##Skills
**Programming Languages:** scientific and general computing in Python, Matlab, 
and IDL

**Scientific Codes:** user for ELITE and BALOO MHD stability codes, data 
preparation and interpretation for EPED pedestal model, developer for EQTOOLS 
Python toolkit

**Experimental Physics:** data analysis, numerical modeling, laser optics, 
X-ray and IR spectroscopy, Thomson Scattering spectrometer systems

**Teaching:** laboratory supervision, review/recitation classroom sessions for 
introductory circuit design for MIT Nuclear Science & Engineering undergraduates

**General:** cross-functional team operations, LaTeX and Adobe Illustrator 
document preparation, SQL database systems