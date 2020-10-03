from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)

# depth = range(1,11)
learning_rates =[x * 0.004 for x in range(1, 61)]
training_accuracy = []
test_accuracy = []
best_train_score = 0
best_test_score = 0

for r in learning_rates:
	#build the model
	model = XGBClassifier(learning_rate = r)

	model.fit(X_train, y_train)
	# clf = KNeighborsClassifier(n_neighbors=n_neighbors)
	# clf.fit(X_train, y_train)
	#record training set accuracy
	training_accuracy.append(model.score(X_train, y_train))
	best_train_score = max(best_train_score,model.score(X_train, y_train))
	#record generalization accuracy
	test_accuracy.append(model.score(X_test, y_test))
	best_test_score = max(best_test_score,model.score(X_test, y_test))


plt.plot(learning_rates, training_accuracy, label="training accuracy")
plt.plot(learning_rates, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Learning Rate")
plt.legend()
plt.show()

print("Accuracy on training set: {:.3f}".format(best_train_score))
print("Accuracy on test set: {:.3f}".format(best_test_score))
