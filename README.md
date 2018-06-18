# Dark Knowledge

This repository contains my work of replicating the Geoffrey Hinton's [work](https://arxiv.org/abs/1503.02531) on distilling the knowledge from a big ensemble network to a smaller neural network.

## Key points
1. To replicate the work, I created a custom optimizer that you can find [here](https://github.com/abhishm/dark_knowledge/blob/master/distill_optimizer.py). This optimizer make sure that the norm of weight going to each individual neuron does not exceed a certain threshold.
2. The main  result of this work is as follwoing:
   1. The missclassification error of the big ensemble network is 94.
   2. The  missclassification error of the smaller network is 123.
   3. The  missclassification error of the smaller network trained on probabilities is 94. 

## Conclusion:
The smaller network was able to extract all the information from the ensemble model when trained on probabilities.

