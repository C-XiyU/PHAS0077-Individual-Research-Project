#!/bin/bash

if [ ! -e clean.inp ]
then
echo 'stream ../toppar.str

open read unit 10 card name ./geom_mutant.psf
read  psf unit 10 card

open read unit 10 card name ./geom_opt.crd
read coor card unit 10 

define struct select .not. (segid SOLV .or. segid POT .or. segid CLA) show end

open write unit 10 card name geom_clean.pdb
write coor unit 10 pdb sele struct end

stop' > clean.inp
fi

~/charmm/charmm < clean.inp > clean.out 

sed -i -e "s#MG#n #g" geom_clean.pdb

~/ADFRsuite-1.0/bin/prepare_receptor -r geom_clean.pdb -o geom_clean.pdbqt

sed -i -e "/n/s/0.000/2.000/" -e "s/n /MG/g" geom_clean.pdbqt

if [ ! -e dock.conf ]
then
echo 'receptor = ./geom_clean.pdbqt
ligand = ../00000001/' > dock.conf
grep PG geom_clean.pdb | awk '{printf("center_x = %8.3f\ncenter_y = %8.3f\ncenter_z = %8.3f\n",$6,$7,$8)}' >> dock.conf
echo 'size_x = 20
size_y = 20
size_z = 20
tasks = 256' >> dock.conf
fi

v=i$1

if [ $v != "idry" ]
then
idock --config dock.conf | tee docking.out
fi
