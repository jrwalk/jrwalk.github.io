---
title: End-To-End Recommender Systems, Part II
layout: post_norel
permalink: /pages/projects/recommender-part-2
description: "Building Recommender systems, part II."
---

In [part I](/pages/projects/recommender-part-1), we discussed building a collaborative-filtering recommender model based on the open-source MovieLens dataset.
Ultimately, we reduced the problem to simple computation based on embedding vectors for our users and movies, such that recommendations could be inferred agnostic of the actual embedding process.
In this post, we'll examine how we can use these embeddings to reliably serve recommendations in a production setting that can scale to large traffic volumes.
