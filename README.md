# Master Thesis - Study on Adversarial Robustness of Phishing Email Detection Model, Code, Data and Results

 Developing robust detection models against phishing emails has long been the main concern of the cyber defense community. Currently, public phishing/legitimate datasets lack
adversarial email examples which keeps the detection models vulnerable. To address this problem, we developed an augmented phishing/legitimate email dataset, utilizing different adversarial text attack techniques. We then detected and analyzed the unique characteristics of the emails that can be easily transformed into adversarial examples. Henceforth, the
models were retrained with an adversarial dataset and the results showed that accuracy and F1 score of the models have been improved from five to forty percent under attack methods.
In another experiment, synthetic phishing emails were generated using a fine-tuned GPT-2 model. The detection model was retrained with a newly formed synthetic dataset and we
have observed the accuracy and robustness of the model did not improve under black box attack methods.
In our last experiment, we proposed a defensive technique to classify adversarial examples to their true labels using K-Nearest Neighbor with 95% accuracy in our prediction.

