#!/bin/bash

echo "Converting all notebooks in current directory to Python script..."

jupyter nbconvert --to python ./*.ipynb
