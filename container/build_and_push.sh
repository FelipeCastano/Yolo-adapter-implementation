#!/bin/bash

docker build -t simplemodel-server -f docker/Dockerfile .
# docker tag simplemodel:latest dockerhub_user/simplemodel:latest
# docker push dockerhub_user/simplemodel:latest
