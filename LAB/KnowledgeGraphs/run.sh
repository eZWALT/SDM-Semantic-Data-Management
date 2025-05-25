docker pull ontotext/graphdb:11.0.1
docker run -d -p 127.0.0.1:7200:7200 -v $(pwd)/data:/opt/graphdb/home --name graphdb -t ontotext/graphdb:11.0.1
docker exec -it graphdb bash


# docker stop graphdb
#docker rm graphdb
