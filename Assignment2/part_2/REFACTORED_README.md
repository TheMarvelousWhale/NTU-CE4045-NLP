# Refactored End-to-end-Sequence-Labeling-via-CNNs-CNNs-CRF-Tutorial

This directory contains the original code (Base_), the experimental code (Dev_) and the final stable code (Refactored_). 
Whereas in the original code, authors use a bilstm, in this modified version we use a CNN, with/without maxpooling, and have 1/3 layers 

The presequisite:
* "./data/glove.6B.100d.txt" is present
* torch is installed

Detailed Explanation:

The Refactored code has moved the data processing functions, the modelling helpers function and the viterbi functions into utils.py
Since we are experimenting with different CNN architectures, we would also declare a list of constant that is the name of the architecture we are going to support. If you want to add more architectures in, append the name to the list and follow the breadcrumbs left in the comments by the author. 

The script has a parameters\['debug_whole_script'\] = True which makes it debug mode, which will do the following:
* rename model to DEBUG_MODEL
* limit epoch to 10
* limit training,dev,set to 1000 e.gs 

turn this off to enable full training

For the implementation wise, we modify the lstm portion of the get_lstm_features by adding a parameter that specify which cnn archi we are using (please rename it in the future to get_features or something -- TODO). Inside this function, which is controlled by the parameters\['word_mode'\], the lstm is swapped out with the cnn, and depends on the archi chosen the corresponding components will be loaded. You should pay attention mainly to;
* get_lstm_features
* class BiLSTM-CRF

The training is done automatically on ALL archi defined. It will also log all results in a text file and save the loss plot for each model. The F1 score can be find in the corresponding .txt file. 

