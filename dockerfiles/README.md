# Tensorflow 1.12 + Python 3.9 images

Official tensorflow 1.12 docker images don't really fit our needs as they depend on python3.8
rather than python 3.9. For this reason two docker files can be find in the `dockerfiles` folder,
one for a cpu image and another for gpu image.

These images live publicly in our [docker hub][kraken_docker].

More info about [docker and tensorflow][tensorflow_dockerfiles]

[kraken_docker]: https://hub.docker.com/r/krakentechnologies/tensorflow/.
[tensorflow_dockerfiles]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/dockerfiles
