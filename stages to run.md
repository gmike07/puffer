### stages to run:
1. train TTP predictor:
    generate train data:
        - simulate X clients, each run under different FCC dataset part, 
            watch a 10 min video. run experiments server with all available algorithms
            without REINFORCE/PUFFER. 
    triain the TTP: run the python script
2. copy the trained TTP, let's call it RTTP (reinforce TTP).
    delete the last layer according the README.
3. train the policy predictor:
    run clients over the training set from the FCC dataset, as in ttp training, ttp now is a constant.
4. test:
    - run the server with puffer and puffer_reinforce
    - run clients over the test set part of FCC
5. results:
    - run the pipeline
    - or either the python script of ssim index plot generator

