import mlflow
import mlflow.sklearn
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt

import dagshub
dagshub.init(repo_owner='Vivek-Kumar-Data', repo_name='mlflow_dagshub_demo', mlflow=True)

# setting the name of the experiment
mlflow.set_experiment('iris_dt')

# adding this code to set our own machine : mlflow tracking server
# this will convert the : file://... ---> http://.....
mlflow.set_tracking_uri("https://dagshub.com/Vivek-Kumar-Data/mlflow_dagshub_demo.mlflow")

# Loading Iris Dataset
iris = load_iris()
x = iris.data
y = iris.target

# Spliting the data using train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# Defining the parameters of the Random Forest Model
max_depth = 1

# From here we will apply mlflow
with mlflow.start_run():
    # calling the model and setting the parameters
    dt = DecisionTreeClassifier(max_depth=max_depth)
    # training the model
    dt.fit(x_train,y_train)
    # predicting the values
    y_pred = dt.predict(x_test)
    # calculating accuracy
    accuracy = accuracy_score(y_test,y_pred)

    # creating confusion matrix
    cm = confusion_matrix(y_test,y_pred)
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")

    # save the plot as artifact
    plt.savefig("confusion_matrix.png")

    # logging the file to save confusion matrix image
    mlflow.log_artifact('confusion_matrix.png')

    # logging code
    mlflow.log_artifact(__file__)

    # logging model
    mlflow.sklearn.log_model(dt,'Decision Tree')

    # logging
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param("max depth",max_depth)

    # just for visualizing printing accuracy
    print("Accuracy : ",accuracy)