FROM pytorch/torchserve:0.5.3-cpu
RUN torchserve --stop
COPY ./model ./model
RUN torch-model-archiver --model-name recom --version 1.0 \
    --model-file /home/model-server/model/model.py \
    --serialized-file /home/model-server/model/recom.pth \
    --export-path /home/model-server/model-store \
    --handler /home/model-server/model/handler.py \
    --extra-files /home/model-server/model/movies_genres.npy,/home/model-server/model/users_age.npy,/home/model-server/model/users_gender.npy
CMD ["torchserve", "--start", "--ncs", "--model-store model-store", "--models recom.mar"]