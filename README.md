# DeepRecommender
A Simple Recommendation System in MovieLens developed with PyTorch and served with TorchServe REST API.


## Dataset
The [MovieLenst-1M](https://grouplens.org/datasets/movielens/) dataset that contains 1 million ratings from 6000 users on 4000 movies

## Model Architecture
The model learns and embedding for every user and movie, combines the embeddings with user features (gender and age) and movie features (genre) to build a rich feature space. On top of that, a 3-layer MLP learns to predict the rating a user would give to a certain movie.

## Results
Result on 10% of the data held out as the validation set: ```RMSE= 0.848```

## How to Run
The trained model and related files are saved in ```./models``` directory.\
\
To use the service, first we need to build a docker image with the following command:
```
$ docker build -t recom .
```

And then run the docker imag:
```
$ docker run -it -p 8080:8080 recom
```

Finally, we can use ```req.py``` to send a request to the service for a ```(user_id, product_id)``` pair, and receive the prediction of the model for the movie's rating by the user. \
\
We can then recommend movies to a user based on these predicted ratings.
