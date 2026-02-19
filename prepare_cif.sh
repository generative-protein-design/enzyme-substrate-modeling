#!/bin/bash


export CIF_FOLDER=$1

parallel -j 4 'mkdir -p /tmp/cif/{%} && cp "$CIF_FOLDER"/*  /tmp/cif/{%}/' ::: $(seq 4)


