#!/bin/sh
cd /home/junoh/2022_DM/adm

reference=$1
sample=$2
log=$3

echo $sample >> $log
output=$(python tools/fid_prdc.py $reference $sample | tail -3)
echo $output >> $log