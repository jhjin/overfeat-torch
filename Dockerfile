FROM bamos/ubuntu-opencv-dlib-torch:ubuntu_14.04-opencv_2.4.11-dlib_19.0-torch_2016.07.12
MAINTAINER Justin Long <crockpotveggies@users.github.com>

ADD . /root/overfeat

RUN /bin/bash /root/overfeat/install.sh
