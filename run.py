import fire

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics
from sklearn.neural_network import MLPClassifier

# The digits dataset
from input import MatrixInput

digits = datasets.load_digits()
n_samples = len(digits.images)


def run(learning_rate, n_training_samples=n_samples // 2):
    if learning_rate <= 0 or learning_rate > 1 or n_training_samples < 1 \
            or n_training_samples > n_samples - 20:
        print('learning-rate must take values (0;1]\n'
              'n-training-samples must take int values [1;%i]' % (n_samples - 20))
        return

    # To apply a classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    data = digits.images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    classifier = MLPClassifier(solver='sgd', activation='logistic', hidden_layer_sizes=(32,),
                               learning_rate='constant', learning_rate_init=learning_rate, verbose=True)

    # We learn the digits on the first half of the digits
    classifier.fit(data[:n_training_samples], digits.target[:n_training_samples])

    # Now predict the value of the digit on the second half:
    expected = digits.target[n_training_samples:]
    predicted = classifier.predict(data[n_training_samples:])

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    def classify(matrix):
        return classifier.predict(matrix)

    matrix_input = MatrixInput()
    matrix_input.show(classify)


def main():
    fire.Fire(run)


if __name__ == '__main__':
    main()
