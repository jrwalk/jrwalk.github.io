# jrwalk.github.io
Portfolio, CV, projects, photos, and bio as a Jekyll site.
Math typesetting in markdown accomplished using [MathJax](https://www.mathjax.org/) distributed network service.

## To install and run:

This repo is dockerized for local development, based on the Debian image.

To build, simply run

```bash
$ docker-compose build
```

then to launch,

```bash
$ docker-compose up jekyll-server
```

which will present the local site on http://localhost:4000.
