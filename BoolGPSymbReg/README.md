# BoolGPSymbReg 

This project contains example of boolean symbolic regression problem solved using ECF framework.
To speed up execution, genetic programs are translated to postfix expression and executed using evaluator implemented in CUDA.

## Build

    chmod +x build.sh
    ./build.sh

## Run

    # Create test dataset:
    cd data/
    python3 input_generator.py > input.txt
    cd ../
    
    # Run executable:
    chmod +x BoolGPSymbReg
    ./BoolGPSymbReg data/parameters.txt
