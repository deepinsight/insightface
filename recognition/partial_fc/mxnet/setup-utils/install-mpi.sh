#!/bin/bash
# install mpi
#wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz

#sudo rm -rf /usr/local/lib/openmpi /usr/local/lib/libmca* /usr/local/lib/libmpi*
#sudo rm -rf /usr/local/lib/libompitrace* /usr/local/lib/libopen* /usr/local/lib/liboshmem* /usr/local/lib/mpi_*


tar zxf openmpi-4.0.0.tar.gz
cd openmpi-4.0.0 || return
sudo ./configure --enable-orterun-prefix-by-default
sudo make -j 48 all
sudo make install
ldconfig