---
title: Synthwave Styling Data Visualizations in Python with Altair
layout: post_norel
permalink: /pages/projects/synthwave-styling-with-altair
---

![synthwave](/images/projects/altair-synthwave/synthwave.jpg)

tl;dr [read more on the Pluralsight tech blog](https://www.pluralsight.com/tech-blog/synthwave-styling-data-visualizations-in-python-with-altair/)

The [Synthwave](https://en.wikipedia.org/wiki/Synthwave) genre of electronic music brings back the aesthetic of 1980s film and music -- and its bright, retro-futuristic style has crossed over into games like _Grand Theft Auto: Vice City_ and _Far Cry 3: Blood Dragon_.
You can even bask in the neon glow of a [VS Code theme](https://marketplace.visualstudio.com/items?itemName=RobbOwen.synthwave-vscode) (and with the glow effect disabled, it's actually a really solid dark mode -- I'm writing this on it right now).
Dominik Haitz's [excellent post on the Matplotlib blog](https://matplotlib.org/matplotblog/posts/matplotlib-cyberpunk-style/) walks through how to apply the style to your visualizations in Python.
But what if we don't want to use Matplotlib?

More recently, a number of alternate visualization libraries have emerged -- [Bokeh](https://docs.bokeh.org/en/latest/), [Plotly](https://plotly.com/dash/), and [Altair](https://altair-viz.github.io/), to name a few, all start from scratch to design a more consistent & powerful visualization experience for Python compared to Matplotlib's often jumbled and haphazard API.
In particular, I've recently started working with Altair, a Python wrapper around the [Vega](https://vega.github.io/vega/) (or more properly, [Vega-Lite](https://vega.github.io/vega-lite/)) declarative visualization grammar.
This means that, in Altair, I simply need to describe _what_ I want the plot to do, in contrast to _how_ to do it (which in Matplotlib generally means jumping through some hidden-state hoops), resulting in a rich, interactive visualization that can be exported to live Javascript, HTML, or static images.

I recently published a post on [Pluralsight's tech blog](https://www.pluralsight.com/tech-blog/synthwave-styling-data-visualizations-in-python-with-altair/) on bringing Synthwave styling to Altair and laying out reusable _themes_ to consistently style all of your plots.
Read more to see how to take a plot from this,

![base](/images/projects/altair-synthwave/01-base.svg)

to this:

![glow](/images/projects/altair-synthwave/08-glow.svg)

Enjoy!
