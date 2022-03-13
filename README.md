# Deep-Recommender
A Simple Recommendation System on MovieLens dataset developed with PyTorch and served with TorchServe REST API.


## Dataset
The [MovieLenst-1M](https://grouplens.org/datasets/movielens/) dataset that contains 1 million ratings from 6000 users on 4000 movies

## Model Architecture
The model learns an embedding for each user and each movie and concatenates the embeddings with user features (gender and age) and movie features (genre) to build a rich feature space. On top of that, a 3-layer MLP learns to predict a user's rating on a certain movie.

## Results
Result on 10% of the data held out as the validation set: ```RMSE= 0.848```

## How to Run
The trained model and related files are located in ```./models``` directory.\
\
To use the service, first we need to build a docker image:
```
$ docker build -t recom .
```

And then run it:
```
$ docker run -it -p 8080:8080 recom
```

Now, we can use ```req.py``` to send a request to the service for a ```(user_id, product_id)``` pair, and receive the prediction of the model for the user's rating on the movie. \
\
We can then recommend movies to the user based on the predicted ratings.
