---
title: Data Processing with Dask
layout: post_norel
permalink: /pages/projects/data-processing-with-dask
---

tl;dr read more [on the Pluralsight Tech Blog](https://www.pluralsight.com/tech-blog/data-processing-with-dask/) or on [Medium](https://medium.com/data-science-and-machine-learning-at-pluralsight/data-processing-with-dask-5e0f6b45b089)

Anymore, it's pretty easy to get a data science / ML task to the point where your own machine can't really keep up with your compute needs.
Rather than shelling out for a single beefy cloud instance, it can sometimes be more worthwhile to think _distributed_ -- by sharding the workout among multiple smaller workers, you can more efficiently grind through big CPU or memory-intensive tasks for large datasets.

Traditionally, this would mean going to something like Apache Spark -- but frankly, Spark is pretty gnarly to develop for, requiring a complete redesign of your workflow without a good way to really locally prototype, and working in it can mean digging through obtuse JVM stacktraces.
With [Dask](https://dask.org/), though, we can bring that compute power to bear in a tool that tightly integrates with our typical Python data tools, like `numpy`, `pandas`, or `scikit-learn`.

I recently published a post on Pluralsight's tech blog for building data pipelines with Dask, available [at the Pluralsight Tech Blog](https://www.pluralsight.com/tech-blog/data-processing-with-dask/) or on [Medium](https://medium.com/data-science-and-machine-learning-at-pluralsight/data-processing-with-dask-5e0f6b45b089).
In it, we:

- process large text files into a usefully structured dataset with a local Dask cluster
- extract useful analytics at scale
- distribute machine learning tasks across the cluster
- see how to kick this up to a fully remote deployed cluster

Enjoy!
