#!/bin/bash

docker run \
  --name mysql-employees \
  -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD=Welcome_1 \
  -v /d/docker-data:/var/lib/mysql \
  genschsa/mysql-employees
