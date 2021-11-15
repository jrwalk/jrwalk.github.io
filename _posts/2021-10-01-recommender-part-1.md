---
title: End-To-End Recommender Systems, Part I
layout: post_norel
permalink: /pages/projects/recommender-part-1
description: "Building Recommender systems, part I."
---

On the Discovery team at Pluralsight, we're responsible for all aspects of our users' content exploration journey.
Some use cases are satisfied by our [core search functionality](https://medium.com/data-science-and-machine-learning-at-pluralsight/an-overview-of-search-at-pluralsight-2fc82173600f), but others require suggesting topics or content to users that don't necessarily know what they're looking for.

## The Data

We'll be using the MovieLens 25M dataset, a collation of twenty-five million movie ratings on a five-star scale collated by the GroupLens team at University of Minnesota, plus genre tags for every reviewed movie.
In our example code, we have bootstrap & migration scripts for representing the movie data at rest in a database - for the purposes of this demo, we can assume simply that the ratings can be accessed easily in a streaming fashion.

## Collaborative Filtering Models

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
