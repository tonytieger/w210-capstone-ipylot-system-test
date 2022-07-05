#!/bin/bash

kubectl delete deployment mosquittoservice
kubectl apply -f mosquittoservice.yaml
