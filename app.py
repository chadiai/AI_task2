import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
    layout="centered"
)

# Define the column names
column_names = ['ID', 'Refractive index', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium','Calcium','Barium','Iron',"Type"]

# Read the CSV file
df = pd.read_csv('glass.data', names=column_names)

st.title("Classification")

mcol1,mcol2 = st.columns([1,5],gap="small")
with mcol1:
        st.write("")
        st.write("SEED: ")
with mcol2:
    randomseed = st.slider("Set the seed number: ", 1, 500, 250,label_visibility='collapsed')


model = st.selectbox("Select a Model", ["Random Forest", "C-Support Vector Classification", "Gradient Boosting"])

if model == "Random Forest" or model == "Gradient Boosting":
    n_estimators = st.slider("Number of Trees", 1, 200, 100)

if model == "C-Support Vector Classification":
    c = st.slider("Number of neighbors", 0, 10000, 1000)


random_seed = randomseed
feature_cols = ['Refractive index', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium','Calcium','Barium','Iron']
X = df[feature_cols]
y = df['Type']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)

if model == "Random Forest":
    clf = RandomForestClassifier(n_estimators=n_estimators)
elif model == "C-Support Vector Classification":
    clf = SVC(C=c)
elif model == "Gradient Boosting":
    clf = GradientBoostingClassifier(n_estimators=n_estimators)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
st.subheader(f"Accuracy: {accuracy}")
cm = confusion_matrix(y_test, y_pred,labels=clf.classes_)
st.subheader("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_).plot()
plt.show()
st.pyplot()
st.header("EDA")

col1,col2 = st.columns(2)
with col1:
    st.markdown("The first 10 rows of the dataset")
    st.write(df.head(10))
with col2:
    st.markdown("The last 10 rows of the dataset")
    st.write(df.tail(10))
with col1:
    st.subheader("Dataset Summary")
    st.write(df.describe())
with col2:
    st.subheader("Dataset Information")
    info_text = f"Number of Rows: {df.shape[0]}\n"
    info_text += f"Number of Columns: {df.shape[1]}\n\n"
    info_text += "Data Types and Non-Null Counts:\n"
    info_text += df.isnull().sum().to_string()
    st.text(info_text)