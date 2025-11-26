import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


iris = load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = y


sns.pairplot(df, hue="species", corner=True)
plt.suptitle("Iris Dataset Pairplot", y=1.02)
plt.show()

df.hist(figsize=(10, 6))
plt.suptitle("Feature Histograms")
plt.show()

df.drop("species", axis=1).plot(kind="box", subplots=True, figsize=(10, 6))
plt.suptitle("Feature Boxplots")
plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

cr = classification_report(y_test, y_pred, target_names=iris.target_names)
print("\nClassification Report:\n", cr)


