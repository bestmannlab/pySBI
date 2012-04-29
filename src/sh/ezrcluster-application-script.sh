#!/bin/bash

cd /tmp
rm -Rf pySBI
rm pySBI.tgz
rm -R /tmp/wta-output
rm -R /tmp/ezrcluster*
scp jbonaiuto@tarsier:/data/projects/pySBI/pySBI.tgz pySBI.tgz
tar xvzf pySBI.tgz

mkdir /tmp/wta-output

export PYTHONPATH=$PYTHONPATH:/tmp/pySBI/src/python
