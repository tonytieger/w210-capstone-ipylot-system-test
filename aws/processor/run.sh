#!/bin/bash
docker run --rm -it --name imageprocessor --network host -v ~/data/:/apps/data/ sriky11/imageprocessor:v1 sh

