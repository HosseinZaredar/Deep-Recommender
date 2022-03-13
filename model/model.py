import torch
import torch.nn as nn

class RatingPredictor(nn.Module):
    
    def __init__(self, n_users, n_genders, n_ages, n_movies, user_emb_dim, movie_emb_dim, n_genres):
        super().__init__()

        self.user_emb = nn.Embedding(n_users, user_emb_dim)
        self.movie_emb = nn.Embedding(n_movies, movie_emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(user_emb_dim+n_genders+n_ages+movie_emb_dim+n_genres, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
    
    def forward(self, users, genders, ages, movies, genres):
        user_embedded = self.user_emb(users)
        movie_embedded = self.movie_emb(movies)
        x = torch.cat((user_embedded, genders, ages, movie_embedded, genres), dim=1)
        x = 6 * self.fc(x)
        return x