# minimal example
### [20 newsgroups dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)
#### one train-example
text

    I was wondering if anyone out there could enlighten me on this car I saw
    the other day. It was a 2-door sports car, looked to be from the late 60s/
    early 70s. It was called a Bricklin. The doors were really small. In addition,
    the front bumper was separate from the rest of the body. This is 
    all I know. If anyone can tellme a model name, engine specs, years
    of production, where this car is made, history, or whatever info you
    have on this funky looking car, please e-mail.
label
    
    rec.autos
    
### two baseline classifiers show almost same performance

[SGD-Classifier](https://scikit-learn.org/stable/modules/sgd.html#sgd)

    train-f1-micro:   0.971
    test-f1-micro:   0.687
    
[MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)

    train-f1-micro:   0.952
    test-f1-micro:   0.688