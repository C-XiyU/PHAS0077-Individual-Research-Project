#!/bin/bash

for i in {1..100} 
do 
m=$(echo $i | awk '{print(($1-1)*50000+1)}') 
o=$(echo $i | awk '{print($1*50000)}') 
n=$(echo $m | awk '{printf("%08d\n",$1)}') 
mkdir -p $n 
obabel version.sdf -f $m -l $o -O $n.sdf
mv $n.sdf $n
done
