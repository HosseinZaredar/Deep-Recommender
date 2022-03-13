import requests

# requesting model's prediction for the rating a user would give to a certain movie 

user_id = 0
movie_id = 1

res = requests.post("http://localhost:8080/predictions/recom", \
    files={'data': f'{user_id},{movie_id}'})

print(res.content.decode('ascii'))