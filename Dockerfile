FROM bamos/ubuntu-opencv-dlib-torch:ubuntu_14.04-opencv_2.4.11-dlib_19.0-torch_2016.07.12
MAINTAINER Justin Long <crockpotveggies@users.github.com>

RUN apt-get update -y; apt-get install wget -y

ADD . /root/overfeat

RUN cd /root/overfeat/; /bin/bash install.sh

CMD /bin/bash
