FROM nvidia/cuda:11.8.0-base-ubuntu20.04

ARG APP=default

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y curl git

RUN curl -fsSL https://pixi.sh/install.sh | sh

ENV PATH="/root/.pixi/bin:${PATH}"

COPY . /app

WORKDIR /app


RUN pixi install -e $APP