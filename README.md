# ml_project_template


## ML Flow
cd ml_project_template
chmod +x model_store/create_mar.sh

docker compose -f container/docker-compose.yml build
docker compose -f container/docker-compose.yml up mlflow-server
docker compose -f container/docker-compose.yml up trainer
docker compose -f container/docker-compose.yml up mar-builder
docker compose -f container/docker-compose.yml up torchserve


docker compose -f container/docker-compose.yml run --rm -it torchserve /bin/bash
docker compose -f container/docker-compose.yml exec -it torchserve /bin/bash

curl http://localhost:8081/models
curl -X POST http://localhost:8080/predictions/simple_model -T data/sample_input.json
find . -type f -name "*Zone.Identifier" -exec rm -f {} +