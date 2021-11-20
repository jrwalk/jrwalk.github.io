---
title: End-To-End Recommender Systems, Part I
layout: post_norel
permalink: /pages/projects/recommender-part-1
description: "Building Recommender systems, part I."
---

On the Discovery team at Pluralsight, we're responsible for all aspects of our users' content exploration journey.
Some use cases are satisfied by our [core search functionality](https://medium.com/data-science-and-machine-learning-at-pluralsight/an-overview-of-search-at-pluralsight-2fc82173600f), but others require suggesting topics or content to users that don't necessarily know what they're looking for.

This is the realm of _recommender models_ -- in this post, we'll examine building a recommender system on a popular open-source dataset using the [LightFM](https://making.lyst.com/lightfm/docs/home.html) package.
In [part II](/pages/projects/recommender-part-2), we'll take that recommendation model and build production-grade model serving infrastructure around it.
Let's dive in!

<> TODO add github link

## The Data

We'll be using the MovieLens 25M dataset, a collation of twenty-five million movie ratings on a five-star scale collated by the GroupLens team at University of Minnesota, plus genre tags for every reviewed movie.
In total, this comprises some 162,000 users rating 62,000 movies.
While this is probably overkill (and MovieLens does provide smaller datasets) this still ends up being tractable even running locally on my laptop and provides a good picture for a production-scale recommender.

In our example code, we have bootstrap & migration scripts for representing the movie data at rest in a database -- for the purposes of this demo, we can assume simply that the ratings can be accessed easily from a datastore (although the specifics of that storage will be important when we start working on serving our recommendations in [part II](/pages/projects/recommender-part-2)).

## Collaborative Filtering Models

Collaborative-filtering models are a bit of an odd duck, at least compared to the model types a beginner machine-learning practitioner might encounter.
Rather than directly predicting a response variable (either continuous for a regression problem or discrete for classification), we're concerned with generating a ranking - that is, given a user and a set of items, we want to assemble the items in that user's predicted preference order.
Recommender models do end up predicting a continuous score, generally intepreted as a similarity metric or distance, but the actual value of this score isn't necessarily as important as whether the scores for different items are correctly ordered relative to each other.

To start, we envision our _interaction matrix_ $$R$$: a large matrix of shape $$n_{users} \times n_{items}$$ containing our raw interaction data (here we use "item" to refer generically to that half of the recommender's input -- in this case, the items are movies).
Each element $$r_{ij}$$ corresponds to the interaction of the $$i$$'th user with the $$j$$'th item.
This is our first branch point in designing recommenders:

(1) _explicit_ interaction, like a user providing a 5-star rating for an item based on their preferences

(2) _implicit_ interaction, where we infer a user's preference -- for example, modeling click-throughs or skips as positive/negative interactions, and treating lack of interaction as neutral or negative

We can think of each row of the matrix as a $$n_{items}$$-sized vector decribing a single user; likewise, each column is a $$n_{users}$$-sized vector describing each item (movie, in this case).
Given vector representations for users and items, then, we can work in terms of similarity (via vector-based distance metrics), which gets at the core of collaborative filtering -- similar users like the same items, and similar items are liked by the same users.
Our recommendation task, then, is to surface the best new items based on a similarity metric: that is, a user is likely to like movies similar to ones they've already enjoyed, or ones that were liked by other users similar to them.

However, we've run into a problem here: in any realistic dataset, any given user will only interact with a tiny fraction of the available inventory of items.
This means that our interaction matrix $$R$$ is extremely sparse (less than a percent non-null entries, in this case), so directly computing vector similarity for our user and item representations is fraught.
Coming from working on a lot of NLP problems, I view this as similar to a bag-of-words problem -- the available vocabulary of tokens is much larger than those that would be used in any reasonable length of text.
Much like an NLP problem, we approach it by learning a dense representation for our users and movies from the sparse interactions.

### Matrix Factorization

Consider our interaction matrix $$R$$.
We wish to generate an approximation of the form

$$
R \approx \hat{R} = UV^T
$$

where $$U$$ is of shape $$n_{users} \times d$$ and $$V$$ is of shape $$n_{items} \times d$$.
In this scheme, the matrices $$U$$ and $$V$$ represent our users and items -- that is, each row of $$U$$ is a $$d$$-dimensional representation of a user, and each row of $$V$$ is a $$d$$-dimensional representation of an item.
Ideally we want to minimize the deviation of our approximation $$\hat{R}$$ from the original $$R$$ such that as much information as possible is retained.

We could do this analytically, e.g. by truncating the results of a singular-value decomposition, but in practice for many collaborative filtering datasets this would be computationally intractible.
Instead, we'll learn our representations from individual interactions $$r_{ij} \in R$$, via

$$
\hat{r}_{ij} = f(\vec{u}_i \cdot \vec{v}_j)
$$

where we learn to predict an individual rating based on a similarity function of the dense user and item vectors (which are simply rows of $$U$$ and $$V$$, taken in a dot product).
Given a sizeable disparity in ratings between users and movies (e.g., a 4-star rating could indicate drastically different degrees of preference for two different users), we usually want to account for a per-user and per-item bias term.
The above then becomes

$$
\hat{r}_{ij} = f(\vec{u}_i \cdot \vec{v}_j + b_i + b_j)
$$

which is the form used in the [original LightFM paper [1]](https://arxiv.org/pdf/1507.08439.pdf).
Several functions are appropriate for $$f(\cdot)$$ here: simply using an identity function allows us to predict numerical ratings, while squashing the vector product through a sigmoid function maps to binary preferences.

This has the added benefit of greatly streamlining training -- rather than needing to work with the entire matrix in memory, we only care about per-element loss $$L(r_{ij}, \hat{r}_{ij})$$, making the problem naturally suited to training via stochastic gradient descent.

### Ranking Loss Functions

In the case of explicit feedback, where our prediction $$\hat{r}_{ij}$$ directly indicates a numerical score or rating, this behaves essentially like a regression problem -- we can learn our embedded representation to minimize an appropriate loss (e.g., mean-squared error)

## Modeling with LightFM

### Sideloading Metadata

## Model Inference

To generate predictions from our model, we don't, in the strictest sense, _need_ to keep the modeling object (LightFM or otherwise) around - in fact, it's often better that we don't!
The model learns representations of users and movies embedded in a low-dimensional space, in which distance (dot-product, in this case) corresponds to the model's score.
Conveniently, LightFM provides an easy way to extract these embeddings (including any sideloaded metadata):

```python
user_bias, user_representation = model.get_user_representations()
movie_bias, movie_representation = model.get_item_representations(
    features=item_features
)

user_embeddings = numpy.concatenate(
    (
        user_representation,
        user_bias.reshape(-1, 1),
        numpy.ones((user_bias.shape[0], 1))
    ),
    axis=1,
)

movie_embeddings = numpy.concatenate(
    (
        movie_representation,
        numpy.ones((movie_bias.shape[0], 1)),
        movie_bias.reshape(-1, 1)
    ),
    axis=1,
)
```

Where we've gone ahead and stacked in the bias vectors to satisfy the dot-product relationship defined for LightFM's scores.
These `(n_users, embedding_dim+2)` and `(n_movies, embedding_dim+2)`-sized matrices include all the information we need to perform inference for known users and movies.
Actually, from this point forward we've essentially abstracted away the details of the process by which we generated these embeddings - from a model-serving standpoint we only need to know that we have embeddings trained for a given dimension and distance metric.

For convenience in the following, suppose we have

```python
Recommendation = tuple[int, float]

user_lookup = ...      # lookup of ordinal index to user ID
movie_lookup = ...     # lookup of ordinal index to movie ID
```

since we'll be accessing the user and movie matrices by ordinal index, we need a lookup table to map those indices to the user and movie IDs in our actual data.

### The Naive Approach: Sorted Scores

We now approach the problem of generating recommendations for a given user - with our embeddings, this is equivalent to taking the $$k$$ largest scores.
Taking the score to represent proximity or distance between points in the embedding space, we arrive at the more common name: "$$k$$ nearest neighbors."

A naive approach to this would be to sort the scores and take the top results, which will return an exact result:

```python
def exact_rank_naive(
    user_index: int, cutoff: int = 10
) -> list[Recommendation]:
    e = user_embeddings[user_index, :]
    scores = numpy.dot(e, movie_embeddings.T)

    ranks = scores.argsort()[:-(cutoff+1):-1]
    return [(movie_lookup[r], scores[r]) for r in ranks]
```

The trouble is, this will require resorting a full array of $$n$$ movies on every request, requiring $$O(n \log n)$$ time over and above the $$O(n)$$ time required to compute every dot product.
For any significant body of items we'd want to recommend, this is excessively slow, even with highly optimized implementations and languages.

### A Better Exact Approach: Partitioning

With a little cleverness, we can optimize this while still computing the exact $$k$$ nearest neighbors.
We can leverage the fact that we only really care about the sorted scores within the top few results to avoid unnecessary computation in the lower-scoring results.
Rather than sorting the whole array, we'll follow a two-step process:

(1) use a selection algorithm ([quickselect](https://en.wikipedia.org/wiki/Quickselect) in this case) to partition the array into the $$k$$ largest scores and $$n-k$$ smallest, without sorting within partitions

(2) sort only the top $$k$$ results for our final recommendations

```python
def exact_rank_partition(
    user_index: int, cutoff: int = 10
) -> list[Recommendation]:
    e = user_embeddings[user_index, :]
    scores = np.dot(e, movie_embeddings.T)
    top_k = np.argpartition(scores, -cutoff)[-cutoff:]

    ranks = np.argsort(scores[top_k])[::-1]
    scores = scores[top_k][ranks]
    top_k = top_k[ranks]
    return [(movie_lookup[k], scores[k]) for k in top_k]
```

The partitioning algorithm on average occupies only $$O(n)$$ time, with the subsequent $$O(k \log k)$$ for step (2) being negligible for $$k \ll n$$.
This nets us a substantial speedup - on my machine, roughly 3-4x faster on the MovieLens data, pushing our inference down to a few milliseconds.
This is good, but we can do better!

### Approximate Nearest Neighbors

In practice, we should be willing to sacrifice a little bit of precision to gain inference speed for our recommendations - after all, we likely won't take much of a hit if we transpose a few results in the top $$k$$, but a system that can't keep up with request volume could represent a much greater loss.

```python
index = annoy.AnnoyIndex(embedding_dim, "dot")
for i, vector in enumerate(movie_embeddings):
    index.add_item(i, vector)
```



```python
def approximate_rank(
    user_index: int, cutoff: int = 10
) -> list[Recommendation]:
    e = user_embeddings[user_index, :]

    ids, scores = index.get_nns_by_vector(
        e, cutoff, include_distances=True
    )
    return [
        (movie_lookup[k], score)
        for k, score in zip(ids, scores)
    ]
```

[1] https://arxiv.org/pdf/1507.08439.pdf
