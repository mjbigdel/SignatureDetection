# SignatureDetection

this project is influence by [Learning Features for Offline Handwritten Signature Verification with CNN](https://arxiv.org/abs/1705.05787) paper.

in the paper they proposed a multi-task feature extractor.

we did Implementation in 3 different way:

  1. [Training Separated CNN as feature extractor, and then using SVMs for classification.](https://github.com/mjbigdel/SignatureDetection/tree/master/Separate%20CNN%20%2B%20SVMs)
  2. [Training shared CNN as feature extractor, and then using SVMs for classification.](https://github.com/mjbigdel/SignatureDetection/tree/master/Shared%20CNN%20%2B%20SVMs)
  3. [Training shared CNN as feature extractor, and then using RNNs (LSTM, GRU, ...) for classification.](https://github.com/mjbigdel/SignatureDetection/tree/master/Shared%20CNN%20%2B%20RNNs)

the documentation and presentation for each part is included.

requirements to run:
  - tensorflow 1.x
  - keras
  - opencv
  
