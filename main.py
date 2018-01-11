

class NaiveBayes:

    def __init__(self):
        self._classes = dict()
        self._attributes = dict()

    def update(self, class_, attributes):
        self._classes[class_] = self._classes.get(class_, 0) + 1
        for i in range(len(attributes)):
            attribute = attributes[i]
            triplet = class_, i, attribute
            self._attributes[triplet] = self._attributes.get(triplet, 0) + 1

    def predict(self, attributes):
        best_score = -1
        best_class = None
        for class_ in self._classes:
            score = self._classes.get(class_, 0)
            for i in range(len(attributes)):
                sum_i_cl = 0
                for cl, i_, attr in self._attributes:
                    if i == i_ and cl == class_:
                        sum_i_cl += self._attributes[cl, i_, attr]
                if sum_i_cl == 0:
                    score = 0
                    continue
                triplet = class_, i, attributes[i]
                score *= self._attributes.get(triplet, 0) / sum_i_cl
            if score > best_score:
                best_score = score
                best_class = class_
        return best_class


def handle(file_, test_len, split_char, print_attr=False):
    classifier = NaiveBayes()
    test = list()
    for line in file_:

        if line[0] == '@' or line[0] == '%' or line == '\n':
            continue
        split = line.split(split_char)
        class_, attributes = split[-1], split[:-1]
        if '?' in attributes:
            continue

        if len(test) < test_len:
            test.append((class_, attributes))
            continue
        else:
            classifier.update(class_, attributes)

    print("Training complete. Test batch:")
    correct = 0
    for class_, attributes in test:
        predicted = classifier.predict(attributes)
        if print_attr:
            print("Attributes: "+str(attributes))
        print("Class: "+class_+"; predicted: "+predicted)
        if class_ == predicted:
            correct += 1
    print("Accuracy: {:.2%}".format(correct/len(test)))

with open("weather.nominal.arff.txt", "r") as f:
    handle(f, 3, ',', True)

with open("soybean.arff.txt", "r") as f:
    handle(f, 83, ', ')
