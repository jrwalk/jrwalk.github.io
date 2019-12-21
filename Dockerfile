FROM debian:latest

WORKDIR /opt/app

RUN \
    apt-get update \
    && apt-get install -y git ruby-full build-essential rubygems zlib1g-dev libxml2

COPY ./Gemfile ./Gemfile
COPY ./Gemfile.lock ./Gemfile.lock

RUN gem install bundler:1.16.1
RUN bundle install
