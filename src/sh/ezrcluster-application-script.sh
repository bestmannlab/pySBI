#!/bin/bash

cd /tmp
rm -Rf pySBI
rm pySBI.tgz
rm -R /tmp/wta-output
scp -P #PORT# #USER#@#HOST#:/data/pySBI/pySBI.tgz pySBI.tgz
tar xvzf pySBI.tgz

mkdir /tmp/wta-output

export PYTHONPATH=$PYTHONPATH:/tmp/pySBI/src/python
