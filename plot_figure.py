from os import listdir
import matplotlib.pyplot as plt
import os

name = "summary_classifier-accuracy.txt"
with open(name, 'r') as f:
    data = f.readlines()

test_acc = []
train_acc = []
loss = []
for line in data:
    # print(line, line[4])
    line = str(line).strip('\n').split(',')
    test_acc.append(float(line[2].strip('\n'))*100)
    train_acc.append(float(line[3].strip('\n'))*100)
    loss.append(float(line[4].strip('\n')))
assert len(test_acc) == len(train_acc)
print(test_acc)

plt.plot(test_acc, label="test accuracy, max_test_accuracy=%0.2f%%" % ((max(test_acc))))
plt.plot(train_acc, label="training accuracy, max_training_accuracy=%0.2f%%" % max(train_acc))

# plt.title("Data Set: Cifar10, Model Type: " + subDir.split("/")[0])
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.legend().set_draggable(True)
plt.show()

plt.plot(loss, label="classification loss")
plt.xlabel("Iterations")
plt.ylabel("Classification Loss")
plt.legend().set_draggable(True)
plt.show()