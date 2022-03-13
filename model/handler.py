import torch
import os
import numpy as np
from model import RatingPredictor

class ModelHandler(object):

    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None
        self.movies_genres = None
        self.users_gender = None
        self.users_age = None

    def initialize(self, context):

        # load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        # load weights
        model = RatingPredictor(6040, 1, 7, 3952, 100, 100, 18)
        model.load_state_dict(torch.load(model_pt_path, map_location=self.device))
        self.model = model

        # load movie and user files
        self.movies_genres = np.load(os.path.join(model_dir, 'movies_genres.npy'))
        self.users_gender = np.load(os.path.join(model_dir, 'users_gender.npy'))
        self.users_age = np.load(os.path.join(model_dir, 'users_age.npy'))

        self.initialized = True


    def handle(self, data, context):
        x = data[0]['data'].decode('ascii')
        x = list(map(int, x.split(',')))

        user = torch.tensor([x[0]], dtype=torch.int)
        gender = torch.tensor([[self.users_gender[user]]], dtype=torch.int)
        age = torch.tensor([self.users_age[user]], dtype=torch.int)
        movie = torch.tensor([x[1]], dtype=torch.int)
        genre = torch.tensor([self.movies_genres[movie]], dtype=torch.int)

        pred_out = self.model(user, gender, age, movie, genre)
        pred_out = pred_out.tolist()[0]
        return pred_out