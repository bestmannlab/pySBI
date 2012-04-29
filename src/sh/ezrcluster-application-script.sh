#!/bin/bash

cd /tmp
scp jbonaiuto@tarsier:/data/projects/pySBI/pySBI.tgz pySBI.tgz
tar xvzf pySBI.tgz

mkdir /tmp/wta-output

export PYTHONPATH=$PYTHONPATH:/tmp/pySBI/src/python
