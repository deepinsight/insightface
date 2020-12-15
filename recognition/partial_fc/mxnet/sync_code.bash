#!/bin/bash

set -xu

file=$1
hosts=$( cat $file | cut -d " " -f 1 )

for host in $hosts
do
	rsync -az ~/insightface* $host:~/insightface
done