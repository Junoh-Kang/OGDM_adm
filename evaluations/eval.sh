#!/bin/sh
cd /home/junoh/2022_DM/adm

reference=$1
sample=$2
log=$3

echo $sample >> $log
output=$(python evaluations/evaluator.py $reference $sample | tail -5)
echo $output >> $log