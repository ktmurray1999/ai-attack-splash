# ai-attack-splash

I made this script to demonstrate how easy it is to fool an AI with noise.

In particular, the AI is a ResNet34 model retrained to identify images. There are only three categories of images (dalmations, soccer balls, and stop signs) and the model is tricked into identifying a stop sign as a dalmation after adding a bit of noise.

The noise is made through backpropagation to optimize for dalmation classification.

The images were borrowed from the Caltech 101 Dataset.