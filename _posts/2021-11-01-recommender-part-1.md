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
Example code for the both parts can be found on Github [here](https://github.com/jrwalk/recommender-demo).
Let's dive in!

## The Data

We'll be using the MovieLens 25M dataset, a collation of twenty-five million movie ratings on a five-star scale collated by the GroupLens team at University of Minnesota, plus genre tags for every reviewed movie.
In total, this comprises some 162,000 users rating 62,000 movies.
While this is probably overkill (and MovieLens does provide smaller datasets) this still ends up being tractable even running locally on my laptop and provides a good picture for a production-scale recommender.

The GroupLens team publishes data in a CSV format -- in particular, we care about `movies.csv`:

```csv
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
3,Grumpier Old Men (1995),Comedy|Romance
4,Waiting to Exhale (1995),Comedy|Drama|Romance
5,Father of the Bride Part II (1995),Comedy
...
```

and `ratings.csv`:

```csv
userId,movieId,rating,timestamp
1,296,5.0,1147880044
1,306,3.5,1147868817
1,307,5.0,1147868828
1,665,5.0,1147878820
1,899,3.5,1147868510
...
```

In our [example code](https://github.com/jrwalk/recommender-demo), we have bootstrap & migration scripts for representing the movie data at rest in a database, along with some sensical transforms (e.g., snake-casing field names, and parsing movie genres into a postgres JSONB array column) -- for the purposes of this demo, we'll assume we can access data in that form.

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

where $$U$$ is of shape $$n_{users} \times d$$ and $$V$$ is of shape $$n_{items} \times d$$, for some comparatively small dimension $$d$$.
In this scheme, the matrices $$U$$ and $$V$$ represent our users and items -- that is, each row of $$U$$ is a $$d$$-dimensional representation of a user, and each row of $$V$$ is a $$d$$-dimensional representation of an item.
Ideally we want to minimize the deviation of our approximation $$\hat{R}$$ from the original $$R$$ such that as much information as possible is retained.

We could do this analytically, e.g. by truncating the results of a [singular-value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition), but in practice for many collaborative filtering datasets this would be computationally intractible.
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

which is the form used in the original LightFM paper, ["Metadata Embeddings for User and Item Cold-start Recommendations"](https://arxiv.org/pdf/1507.08439.pdf) by Maciej Kula.
Several functions are appropriate for $$f(\cdot)$$ here: simply using an identity function allows us to predict numerical ratings, while squashing the vector product through a sigmoid function maps to binary preferences.

This has the added benefit of greatly streamlining training -- rather than needing to work with the entire matrix in memory, we only care about per-element loss $$L(r_{ij}, \hat{r}_{ij})$$, making the problem naturally suited to training via stochastic gradient descent.

### Ranking Loss Functions

In the case of explicit feedback, where our prediction $$\hat{r}_{ij}$$ directly indicates a numerical score or rating, this behaves essentially like a regression problem -- we can learn our embedded representation to minimize an appropriate loss (e.g., mean-squared error), which is well-suited for learning by stochastic gradient descent or alternating least squares.

Things get more interesting when we are dealing with implicit feedback, e.g., when we only have binary interaction (or lack thereof) rather than explicit ratings to predict.
In the explicit case, we could leverage high versus low ratings to teach the model to distinguish preferences.
In the implicit case, however, we can't distinguish unexamined versus disliked items.
Rather than treating unexamined items as having a zero rating (and therefore being forced to the bottom of the stack), we care about a somewhat more relaxed criterion -- that they only be ranked appropriately compared to known positive interactions.

LightFM supports two common approaches to this: [Bayesian Personalized Ranking (BPR)](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) and [Weighted Approximate-Rank Pairwise (WARP) loss](http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf).
In both cases, rather than simply computing the loss for a (user, item) pair, we work with a triple: a user, a positive item, and a negative item (actually disliked, or merely unexamined).
Our loss then centers on whether the model appropriately ranks the positive and negative item relative to each other.

In BPR loss, we randomly sample a positive and negative sample, score both, and take the difference between the positive and negative samples' scores.
This is then passed through a sigmoid function to squash the value onto $$(0, 1)$$, which is interpreted as the probability the user actually prefers the positive sample over the negative.
This is then used as a loss to update the user and item representations via stochastic gradient descent.
Conceptually, this loss scheme ends up optimizing for ROC-AUC in our ranking, since the receiver-operating characteristic is directly tied to a likelihood of ranking a randomly-selected positive sample over a random negative sample.

In WARP loss, we begin similarly, by sampling a (user, positive item, negative item) triple at random.
However, rather than a sigmoid loss on the relative scores, we use a hinge loss: we skip the update entirely if the positive and negative samples are correctly relatively ranked, and only update our weights in rank-violating cases.
Moreover, if we rank a pair correctly, we draw another random negative sample until we find a violating case, for which we update our weights.
This introduces a subtle shift in the optimization: rather than correctly ranking a randomly-selected positive sample versus a randomly selected negative, the algorithm tries to correctly rank against _any_ negative sample.
This tends to optimize towards correctly ranking the top few examples rather than deeper in the stack, tending towards better precision-at-$$k$$ rather than AUC.
It does, however, introduce a quirk to the training process -- while early epochs run relatively quickly, as the model trains it will need to sample more negative examples before finding a rank-violating pair, increasing training time (usually we set a maximum sample count as a model hyperparameter).

## Modeling with LightFM

### Sideloading Metadata

### Building the model

First, we need to assemble our dataset:

```python
from lightfm.data import Dataset

# assuming we've got an open DB connection to our datastore
users = conn.execute("select distinct(user_id) from ratings;")
movies = conn.execute("select id from movies;")
genres = conn.execute(
    "select distinct(jsonb_array_elements(genres)) from movies;"
)

dataset = Dataset()
dataset.fit(users, movies, item_features=genres)
```

Contrary to the name, this isn't fitting a model: rather, all this builds is an internal mapping of users, items, and user/item features (movies + movie genre tags, in this case) to ordinal indices for the sparse matrix representation used by the model, so it just needs sequences of the unique IDs and discrete features.

Next, we build our interactions:

```python
top_rated = conn.execute(
    "select user_id, movie_id from ratings where rating = 5.0;"
)
interactions, weights = dataset.build_interactions(top_rated)
```

Note that we actually output two matrices here: while both correspond in size to our matrix $$R$$ (that is, it's $$n_{users} \times n_{items}$$), `interactions` stores only binary on/off states for each interaction $$r_{ij}$$, while `weights` stores a weight of the corresponding interaction, which can be either explicitly supplied to `build_interactions` or computed from repeated `(user_id, movie_id)` interactions.
To recover the matrix $$R$$ with numerical ratings (e.g., when we care about numerical score, or want to use repeated interaction to indicate stronger preference), we simply multiply the two matrices element-wise.
In this case, we actually only care about the binary state, so we can safely discard the `weights` matrix.

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
