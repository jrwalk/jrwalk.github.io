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

We'll be using the [MovieLens 25M](https://grouplens.org/datasets/movielens/) dataset, a collation of twenty-five million movie ratings on a five-star scale collated by the GroupLens team at University of Minnesota, plus genre tags for every reviewed movie.
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

Collaborative-filtering models are a bit of an odd duck, at least compared to the forms a beginner machine-learning practitioner might encounter.
Rather than directly predicting a response variable (either continuous for a regression problem or discrete for classification), we're concerned with generating a ranking - that is, given a user and a set of items, we want to assemble the items in that user's predicted preference order.
Recommender models do end up predicting a continuous score, generally intepreted as a similarity metric or distance, but the actual value of this score isn't necessarily as important as whether the scores for different items are correctly ordered relative to each other.

To start, we envision our _interaction matrix_, denoted $$R$$: a large matrix of shape $$n_{users} \times n_{items}$$ containing our raw interaction data (here we use "item" to refer generically to that half of the recommender's input -- in this case, the items are movies).
Each element $$r_{ij}$$ corresponds to the interaction of the $$i$$'th user with the $$j$$'th item.
The nature of this interaction is our first branch point in designing recommenders:

(1) _explicit_ feedback, like a user providing a 5-star rating for an item based on their preferences

(2) _implicit_ feedback, where we infer a user's preference -- for example, modeling click-throughs or skips as positive/negative interactions, and treating lack of interaction as neutral or negative

We can think of each row of the matrix as a $$n_{items}$$-sized vector decribing a single user; likewise, each column is a $$n_{users}$$-sized vector describing each item (movie, in this case).
Given vector representations for users and items, then, we can work in terms of similarity (via vector-based distance metrics), which gets at the core of collaborative filtering -- similar users like the same items, and similar items are liked by the same users.
Our recommendation task, then, is to surface the best new items based on a similarity metric: that is, a user is likely to prefer movies similar to ones they've already enjoyed, or ones that were liked by other users similar to them.

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
In this scheme, the matrices $$U$$ and $$V$$ represent our users and items -- that is, each row of $$U$$ is a $$d$$-dimensional embedded representation of a user, and each row of $$V$$ is a $$d$$-dimensional embedded representation of an item.
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

This has the added benefit of greatly streamlining training -- rather than needing to work with the entire matrix in memory, we only care about per-element loss $$L(r_{ij}, \hat{r}_{ij})$$, making the problem naturally suited to training via [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).

### Ranking Loss Functions

In the case of explicit feedback, where our prediction $$\hat{r}_{ij}$$ directly indicates a numerical score or rating, this behaves essentially like a regression problem -- we can learn our embedded representation to minimize an appropriate loss (e.g., mean-squared error), which is well-suited for learning by SGD or alternating least squares.

Things get more interesting when we are dealing with implicit feedback, e.g., when we only have binary interaction (or lack thereof) rather than explicit ratings to predict.
In the explicit case, we could leverage high versus low ratings to teach the model to distinguish preferences.
In the implicit case, however, we can't distinguish unexamined versus disliked items.
Rather than treating unexamined items as having a zero rating (and therefore being forced to the bottom of the stack), we care about a somewhat more relaxed criterion -- that they only be ranked appropriately compared to known positive interactions.

LightFM supports two common approaches to this: [Bayesian Personalized Ranking (BPR)](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) and [Weighted Approximate-Rank Pairwise (WARP) loss](http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf).
In both cases, rather than simply computing the loss for a (user, item) pair, we work with a triple: a user, a positive item, and a negative item (actually disliked, or merely unexamined).
Our loss then centers on whether the model appropriately ranks the positive and negative item relative to each other.

In BPR loss, we randomly draw a positive and negative sample, score both, and take the difference between the positive and negative samples' scores.
This is then passed through a sigmoid function to squash the value onto $$(0, 1)$$, which is interpreted as the probability the user actually prefers the positive sample over the negative.
This is then used as a loss to update the user and item representations via stochastic gradient descent.
Conceptually, this loss scheme ends up optimizing for [ROC-AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) in our ranking, since the receiver-operating characteristic is directly tied to a likelihood of ranking a randomly-selected positive sample over a random negative sample.

In WARP loss, we begin similarly, by sampling a (user, positive item, negative item) triple at random.
However, rather than a sigmoid loss on the relative scores, we use a hinge loss: we skip the update entirely if the positive and negative samples are correctly relatively ranked, and only update our weights in rank-violating cases.
Moreover, if we rank a pair correctly, we draw another random negative sample until we find a violating case, for which we update our weights.
This introduces a subtle shift in the optimization: rather than correctly ranking a randomly-selected positive sample versus a randomly selected negative, the algorithm tries to correctly rank against _any_ negative sample.
This tends to optimize towards correctly ranking the top few examples rather than deeper in the stack, tending towards better precision-at-$$k$$ rather than AUC.
It does, however, introduce a quirk to the training process -- while early epochs run relatively quickly, as the model trains it will need to sample more negative examples before finding a rank-violating pair, increasing training time (usually we set a maximum sample count as a model hyperparameter).
The algorithm also incorporates this feedback into its learning rate, treating an elevated number of negative samples before finding a rank-violating pair as indicative that the model is near-optimum, and slowing the learning rate accordingly.

### Sideloading Metadata

Before we begin, there is one more factor to consider.
The Achilles' heel of any recommender system is the "cold-start problem" -- that is, how to generate recommendations for users/items that have few or no interactions.
A purely collaborative-filtering approach, like we described above, is unable to infer meaningful scores in this case.
Content-based recommenders, which try to predict user preferences based on explicit features (more like a conventional regression model) are able to generate predictions from known metadata, e.g. item tags, but these cannot leverage the collaborative wisdom of our interaction data (which often can be more informative and reliable than users' stated preferences!).

LightFM implements a [clever hybrid approach](https://arxiv.org/pdf/1507.08439.pdf) for sideloading user/item features along with the main collaborative-filtering model, boosting performance for items with little interaction data.
Consider a user or item as having some number of discrete features (i.e., a metadata tag) associated with them.
When a user interacts with an item, we record them as also having interacted with all that item's features, and vice versa for user features.
In the sparse view, this is analogous concatenating an $$n_{user} \times n_{item-feature}$$ or $$n_{user-feature} \times n_{item}$$ interaction matrix to the primary data.
These features then have their own learned dense representations in the model, with the final embedded representation of a user or item being the linear combination of its own vector and those of its features.
Alternately, we could simply consider each user/item as having a self-referential "identity" feature as well as the sideloaded metadata, and treat its representation as the average of its feature vectors.
In the case where no user or item features are present (i.e., each is represented only by its identity feature) then the problem reduces to traditional collaborative filtering.


This has the effect of allowing users/items with poor presence in the interaction data to bootstrap their representation based on interactions of other users/items with shared features.
Again, we can draw a parallel to NLP problems -- this approach is conceptually similar to the subword embeddings used by [FastText](https://fasttext.cc/), which represents words as the average of the token embedding and embeddings of its constituent substrings, compensating for rare or out-of-vocabulary words by building nontrivial representations from the substring embeddings.
Of course, poor or uninformative features can end up muddling the learned representation of items with a large number of interactions, and a uneven distribution of tags can similarly add noise to the model rather than improving it.
However, in cases with fairly uniform feature coverage, it can improve model performance, particularly for the underrepresented "long tail" of user-item interactions.

## Modeling with LightFM

We're now ready to begin tackling generating recommendations from our MovieLens data.
Although our data includes explicit ratings (on a 5-star scale, in half-star increments), user ratings are notoriously fickle -- even accounting for per-user and per-movie bias, the star rating can be a noisy & unreliable representation of a user's preference.
However, we can easily smooth our rating data into something that looks more like the implicit case, by simply treating the users' 5-star ratings as a positive signal and anything else as negative.
We'll train our model using WARP loss, to optimize specifically for precision-at-$$k$$, aiming to get the top few rankings correct.
Combined with our restriction to 5-star interactions, this should learn to generate a few high-quality recommendations based on the users' most highly-preferred movies, discarding the vast majority of the 62,000+ movies in the dataset as irrelevant.

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

Contrary to the name, this isn't fitting a model: rather, all this builds is an internal mapping of users, items, and user/item features (movies + twenty genre tags, in this case) to ordinal indices for the sparse matrix representation used by the model, so it just needs sequences of the unique IDs and discrete features.

Next, we build our interactions:

```python
five_star_ratings = conn.execute(
    "select user_id, movie_id from ratings where rating = 5.0;"
)
interactions, weights = dataset.build_interactions(five_star_ratings)
```

Note that we actually output two matrices here: while both correspond in size to our matrix $$R$$ (that is, it's $$n_{users} \times n_{items}$$), `interactions` stores only binary on/off states for each interaction $$r_{ij}$$, while `weights` stores a weight of the corresponding interaction, which can be either explicitly supplied to `build_interactions` or computed from repeated `(user_id, movie_id)` interactions.
To recover the matrix $$R$$ with numerical ratings (e.g., when we care about numerical score, or want to use repeated interaction to indicate stronger preference), we simply multiply the two matrices element-wise.
In this case, we actually only care about the binary state (and each user only rates a given movie once), so we can safely discard the `weights` matrix.
Examining the interactions matrix, we see

```python
<162541x62423 sparse matrix of type '<class 'numpy.int32'>'
	with 3612474 stored elements in COOrdinate format>
```

meaning our model is extremely sparse: less than a tenth of a percent of possible interactions are actually populated.

For validation purposes, we can split this into a training and test set:

```python
from lightfm.cross_validation import random_train_test_split

train, test = random_train_test_split(interactions, test_percentage=0.2, random_state=42)
```

Each of these interaction matrices is of the same size as our original: rather than splitting users or movies out, we randomly select individual interactions to remove for the test set, such that the model is trained on a masked subset of user/item interactions.
Effectively, the test metric becomes "given 80% of a user's interactions, how well can we predict the last 20%?".
This makes no guarantees that users or items will be well-represented in either split -- for very sparse data, it is possible that a given user or movie may have few or no interactions in the test set.
However, in general this gives us a good picture of the model's performance on unseen interactions for known users.

Next, we build our sideloaded movie features:

```python
movie_genres = conn.execute("select id, genres from movies;")
item_features = dataset.build_item_features(movie_genres)
```

Examining this matrix, we see

```python
<62423x62443 sparse matrix of type '<class 'numpy.float32'>'
	with 174730 stored elements in Compressed Sparse Row format>
```

Strictly, this is a selection matrix for item features: without any sideloaded features, this would simply be a $$62423 \times 62423$$ identity matrix.
The additional 20 columns contain indicators for the 20 distinct genre tags that can be appended to a movie, such that the vector representation of a movie is composed of a linear combination of its self-referential "identity" feature and its genre vectors, all of which are learned during training time.

As discussed above, sideloaded features are most useful in the "long tail" of users/items with poor representation in the interaction data.
In the case of our MovieLens data, we're dealing with just such a distribution:

<p align="center">
  <img src="/images/projects/recommender/rating_counts_log.svg" />
</p>


The top 1000 movies (out of 62,000+) by rating count comprise about 60% of the total rating count, while the top 4000 comprise ~90%.
Since we have fairly uniform coverage of genre tags for these movies, I'd expect we see some performance gain particularly in the tail.
Intuitively, the feature embeddings of genre tags will tend to drag movies towards their genre centroids in the embedding space, such that users with strong genre preference will still be relatively close even to movies with poor representation in the interaction data.

We're now ready to train our model:

```python
from lightfm import LightFM

num_threads = ...  # however many cores you want to devote to it!

model = LightFM(loss="warp", no_components=64)
model.fit(
    train, item_features=item_features, epochs=250, num_threads=num_threads
)
```

Alternately, we could warm-start the training on an existing method with the `fit_partial` method, which is useful for checkpointing during training.
Note, however, that the scoring functions are comparatively slow, as they require computing and then averaging ranking metrics per-user across the entirety of the movie catalog.
Model training is parallelized via [HOGWILD SGD](https://arxiv.org/abs/1106.5730), meaning we reap substantial benefits from throwing more CPU cores at our training process.

LightFM provides built-in scoring functions for common ranking metrics:

```python
from lightfm.evaluation import precision_at_k, auc_score

train_p = precision_at_k(
    model, train, k=5, item_features=item_features, num_threads=num_threads
).mean()
test_p = precision_at_k(
    model, validate, k=5, item_features=item_features, num_threads=num_threads
).mean()
train_auc = auc_score(
    model, train, item_features=item_features, num_threads=num_threads
).mean()
test_auc = auc_score(
    model, validate, item_features=item_features, num_threads=num_threads
).mean()

print(f"training p@5: {train_p:.3f}")
print(f"testing p@5: {test_p:.3f}")
print(f"training AUC: {train_auc:.3f}")
print(f"testing AUC: {test_auc:.3f}")
```

yielding

```python
training p@5: 0.352
testing p@5: 0.070
training AUC: 0.999
testing AUC: 0.991
```

As expected for our sparse data, the sheer number of negative samples compared to positive ensures even a mediocre model will hit a high AUC.
Similarly, for how few positive samples we have, this is an acceptable precision at a cutoff of 5.
Notably, we see a respectable gain in precision over a pure CF model by including our genre data -- 5.3% without sideloaded metadata, compared to 7.0% with the metadata, although interestingly the training precision was substantially higher .

Outside the true positives, the recommendation quality seems fairly good.
Even for a user with only two ratings, we find both true positives (indicated by `**`) in the top 10:

```csv
*Secret Garden, The (1993)*
Harriet the Spy (1996)
Matilda (1996)
Babe (1995)
Little Princess, A (1995)
Madeline (1998)
*...And God Spoke (1993)*
Trouble with Angels, The (1966)
Pollyanna (1960)
It's a Very Merry Muppet Christmas Movie (2002)
```

while the others are consistently similar children's movies.
With more ratings, we still see sensical results:

```csv
*Lord of the Rings: The Two Towers, The (2002)*
Lord of the Rings: The Fellowship of the Ring, The (2001)
*Inception (2010)*
Matrix, The (1999)
*Shawshank Redemption, The (1994)*
Lord of the Rings: The Return of the King, The (2003)
*Fight Club (1999)*
*Memento (2000)*
*Forrest Gump (1994)*
*Green Mile, The (1999)*
```

The results are either true positives, or pass gut-check (e.g., enjoying all three films of the _Lord of the Rings_ trilogy despite only having rated one).

## Model Inference

To generate predictions from our model, we don't, in the strictest sense, _need_ to keep the modeling object around (though LightFM provides a `predict` method that will score user-item combinations easily) - in fact, it's often better that we don't!
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

Since we'll be accessing the user and movie matrices by ordinal index, we also need a lookup table to map those indices to the user and movie IDs in our actual data -- ortunately, we can easily extract this from the `lightfm.data.Dataset` object by

```python
# discard user feature mapping, as we didn't use any
user_lookup, _, movie_lookup, movie_feature_lookup = dataset.mapping()
```

since we'll be accessing the user and movie matrices by ordinal index, we need a lookup table to map those indices to the user and movie IDs in our actual data -- fortunately, we can easily extract this from the `lightfm.data.Dataset` object.

For convenience in the following, suppose we have

```python
Recommendation = tuple[int, float]
```

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
This is the realm of _approximate nearest-neighbor search_: methods that sacrifice exactly-correct nearest neighbor retrieval (though they're often pretty close!) for significantly faster search times.
Generally, these methods function by reducing the search into a two-step process: a coarse search that can rapidly reduce the search space to comparatively few candidates, and a refinement step that then finds the best-scoring candidates from within the reduced set (our partitioning method above also follows this scheme -- however, ANN methods can execute the coarse step in sub-linear time).

A number of efficient implementations exist, using various approaches for the coarse search step, including [nmslib](https://github.com/nmslib/nmslib) for [navigable-small-world graphs](https://arxiv.org/abs/1603.09320), Facebook's [FAISS](https://github.com/facebookresearch/faiss), and Spotify's [annoy](https://github.com/spotify/annoy).
I'm particularly fond of annoy, for several reasons -- it's fast and accurate (though HNSW graphs are the current state-of-the-art for retrieval metrics, e.g. precision or recall at $$k$$), but it also makes a machine-learning engineer's life easier!
Annoy indices are written simply as memory-mapped static files, making them trivial to load into (or share across) processes, and even enabling lazy-loading components of the index as needed from disk on machines with constrained memory.
This makes annoy indices remarkably simple to work with in the context of scalable model-serving APIs, whereas something like nmslib requires specialized instances provisioned with significant compute resources.

To understand how annoy approaches coarse search, let's consider points in a $$d$$-dimensional embedding space (for the purposes of visualization, we'll create a toy 2D space):

<p align="center">
  <img src="/images/projects/recommender/annoy-1.svg" />
</p>

Next, draw a hyperplane through the space at random.
It is a trivial vector operation to establish which side of the hyperplane each point lies on:

<p align="center">
  <img src="/images/projects/recommender/annoy-2.svg" />
</p>

At this point, we've essentially built an extremely simple hashing function: we can map an arbitrary point in a $$d$$-dimensional space to a single bit indicating the sides of the hyperplane.
By ordinary hash function standards, this isn't particularly helpful, as we have a huge number of hash collisions.
However, those collisions have one useful property: the likelihood of collision between two points is dependant on their proximity, which we term [locality-sensitive hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing).

Next, take the two subspaces defined by our hyperplane, and subdivide them with two further random hyperplanes (in annoy's case, we specifically select planes by splitting evenly between two randomly selected points in the dataset, to avoid particularly pathological splits):

<p align="center">
  <img src="/images/projects/recommender/annoy-3.svg" />
</p>

By recursively splitting our space in this manner, we can create a set of shards, each of which contains points that are comparatively close (by our chosen distance metric).
Each of the splits is stored as a node within a binary search tree -- for a new point, we simply traverse the tree by evaluating the point against that node's hyperplane and taking the appropriate branch.
Thus, we can match an embedding (the "query" point) to a shard containing good candidates for nearest-neighbors, with the number of computations determined by the splitting depth of the tree, rather than the total number of points, allowing us to execute the coarse search step in only a handful of computations.

We do, however, need to account for errors introduced by the splitting boundaries, e.g. when a query point lies very near the split such that a large number of valid results in the neighboring shard would be rejected.
At index time, we can correct for errors by building a forest of multiple trees with independent splits, and taking the ensemble vote of their results as the candidate set to avoid rejections due to a single bad split -- however, this inflates the size of the index.
At query time, annoy will also inspect neighboring shards in the trees, rather than just the final node, to generate candidates -- this is available as a request parameter, higher values of which will result in longer query times but more accurate results.

In code, building an annoy index is straightforward:

```python
index = annoy.AnnoyIndex(movie_embeddings.shape[1], "dot")
for i, vector in enumerate(movie_embeddings):
    index.add_item(i, vector)

index.build(num_trees, n_jobs=n_threads)
index.save("/path/to/index.ann")
```

where we specify an index expecting embeddings of the correct dimension, and using the a dot-product metric to match the expected distance function from LightFM.
We then build the index with our selected number of trees, which can be done in parallel -- this should be as large as we can reasonably accept for disk/memory requirements, so requires some manual tuning.
I've had good luck with ~100 trees in the index.
Saving then writes this to a memory-mapped file on disk, which can be passed around to different worker processes and loaded by simply calling

```python
# must match what we created it with!
index = annoy.AnnoyIndex(embedding_dim, "dot")
index.load("/path/to/index.ann", prefault=False)
```

Note that, by default, the index will lazy-load by pages into memory as needed, but setting `prefault=True` will load the entire index.
The index provides nearest-neighbor lookup by index (for items already in the index) or by vector (for an arbitrary query embedding), which we can use in a similar manner to our exact rank functions above:

```python
def approximate_rank(
    user_index: int, cutoff: int = 10
) -> list[Recommendation]:
    e = user_embeddings[user_index, :]

    ids, scores = index.get_nns_by_vector(
        e, cutoff, include_distances=True,
    )
    return [
        (movie_lookup[k], score)
        for k, score in zip(ids, scores)
    ]
```

On my machine, this returns nearest neighbors in sub-millisecond times, even hitting below one hundred microseconds for smaller queries, while still returning reasonable results for its recommendations.

## Next Steps

With that, we have a good solution for quickly generating recommendations from our model -- we can generate and store embeddings for our users and movies, as well as a lookup table of user/movie IDs to their ordinal indices in the embeddings (provided by the `lightfm.data.Dataset` object via its `mappings` method), and build an approximate-nearest-neighbors index in Annoy of the movie embeddings.
With these artifacts, we're ready to go to the next step: [building the serving infrastructure for a recommender API](/pages/projects/recommender-part-2).
