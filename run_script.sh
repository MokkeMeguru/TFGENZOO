#!/bin/sh

docker-compose up -d
docker-compose exec tfgenzoo /bin/zsh
