import streamlit as st
import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from keras.models import Sequential
# from keras.models import load_model

from keras.layers import Dense

from datetime import datetime

st.set_page_config(page_title="Predicting Smart Grid Stability with Deep Learning", page_icon=":zap:", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)

st.sidebar.title('Predicting Smart Grid Stability with Deep Learning')
st.sidebar.write("Source: [Paolo Breviglieri](https://www.kaggle.com/pcbreviglieri/predicting-smart-grid-stability-with-deep-learning/notebook)")

st.sidebar.header('Problem Statement')
st.sidebar.write('(explain what the project is about)')

DATE_COLUMN = 'date/time'
DATA_URL = ('smart_grid_stability_augmented.csv')

@st.cache
def load_data():
    sns.set()
    start_time = datetime.now()
    data = pd.read_csv(DATA_URL)
    # lowercase = lambda x: str(x).lower()
    # data.rename(lowercase, axis='columns', inplace=True)
    # data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])

    map1 = {'unstable': 0, 'stable': 1}
    data['stabf'] = data['stabf'].replace(map1)
    data = data.sample(frac=1)

    X_training = pd.read_csv('X_training.csv', index_col=0)
    y_training = pd.read_csv('y_training.csv', index_col=0)
    X_testing = pd.read_csv('X_testing.csv', index_col=0)
    y_testing = pd.read_csv('y_testing.csv', index_col=0)

    return data, X_training, y_training, X_testing, y_testing

st.header('Dataset')
data_load_state = st.text('Loading data...')
data, X_training, y_training, X_testing, y_testing = load_data()
data_load_state.text("Dataset successfully loaded!")

st.write('The dataset chosen for this machine learning exercise has a synthetic nature and contains results from simulations of grid stability for a reference 4-node star network, as described in 1.2.')
st.write('The original dataset contains 10,000 observations. As the reference grid is symetric, the dataset can be augmented in 3! (3 factorial) times, or 6 times, representing a permutation of the three consumers occupying three consumer nodes. The augmented version has then 60,000 observations. It also contains 12 primary predictive features and two dependent variables.')

st.write(data.head())

st.subheader('Predictive features:')
st.write("- 'tau1' to 'tau4': the reaction time of each network participant, a real value within the range 0.5 to 10 ('tau1' corresponds to the supplier node, 'tau2' to 'tau4' to the consumer nodes)")
st.write("- 'p1' to 'p4': nominal power produced (positive) or consumed (negative) by each network participant, a real value within the range -2.0 to -0.5 for consumers ('p2' to 'p4'). As the total power consumed equals the total power generated, p1 (supplier node) = - (p2 + p3 + p4)")
st.write("- 'g1' to 'g4': price elasticity coefficient for each network participant, a real value within the range 0.05 to 1.00 ('g1' corresponds to the supplier node, 'g2' to 'g4' to the consumer nodes; 'g' stands for 'gamma')")

st.subheader('Dependent variables:')

st.write("- 'stab': the maximum real part of the characteristic differentia equation root (if positive, the system is linearly unstable; if negative, linearly stable)")
st.write("- 'stabf': a categorical (binary) label ('stable' or 'unstable').")

st.write("As there is a direct relationship between 'stab' and 'stabf' ('stabf' = 'stable' if 'stab' <= 0, 'unstable' otherwise), 'stab' will be dropped and 'stabf' will remain as the sole dependent variable.")
st.write('As the dataset content comes from simulation exercises, there are no missing values. Also, all features are originally numerical, no feature coding is required. Such dataset properties allow for a direct jump to machine modeling without the need of data preprocessing or feature engineering.')


if st.checkbox('Show raw data'):
    st.subheader('Shuffled dataframe')
    st.write(data)

def assessment(f_data, f_y_feature, f_x_feature, f_index=-1):
    """
    Develops and displays a histogram and a scatter plot for a dependent / independent variable pair from
    a dataframe and, optionally, highlights a specific observation on the plot in a different color (red).

    Also optionally, if an independent feature is not informed, the scatterplot is not displayed.

    Keyword arguments:

    f_data      Tensor containing the dependent / independent variable pair.
                Pandas dataframe
    f_y_feature Dependent variable designation.
                String
    f_x_feature Independent variable designation.
                String
    f_index     If greater or equal to zero, the observation denoted by f_index will be plotted in red.
                Integer
    """
    for f_row in f_data:
        if f_index >= 0:
            f_color = np.where(f_data[f_row].index == f_index,'r','g')
            f_hue = None
        else:
            f_color = 'b'
            f_hue = None

    f_fig, f_a = plt.subplots(2, 1, figsize=(8,8))

    f_chart1 = sns.distplot(f_data[f_x_feature], ax=f_a[0], kde=False, color='g')
    f_chart1.set_xlabel(f_x_feature,fontsize=10)

    if f_index >= 0:
        f_chart2 = plt.scatter(f_data[f_x_feature], f_data[f_y_feature], c=f_color, edgecolors='w')
        f_chart2 = plt.xlabel(f_x_feature, fontsize=10)
        f_chart2 = plt.ylabel(f_y_feature, fontsize=10)
    else:
        f_chart2 = sns.scatterplot(x=f_x_feature, y=f_y_feature, data=f_data, hue=f_hue, legend=False)
        f_chart2.set_xlabel(f_x_feature,fontsize=10)
        f_chart2.set_ylabel(f_y_feature,fontsize=10)

    # plt.show()
    st.pyplot(plt)


def correlation_map(f_data, f_feature, f_number):
    """
    Develops and displays a heatmap plot referenced to a primary feature of a dataframe, highlighting
    the correlation among the 'n' mostly correlated features of the dataframe.

    Keyword arguments:

    f_data      Tensor containing all relevant features, including the primary.
                Pandas dataframe
    f_feature   The primary feature.
                String
    f_number    The number of features most correlated to the primary feature.
                Integer
    """

    f_most_correlated = f_data.corr().nlargest(f_number,f_feature)[f_feature].index
    f_correlation = f_data[f_most_correlated].corr()

    f_mask = np.zeros_like(f_correlation)
    f_mask[np.triu_indices_from(f_mask)] = True
    with sns.axes_style("white"):
        f_fig, f_ax = plt.subplots(figsize=(20, 10))
        sns.heatmap(f_correlation, mask=f_mask, vmin=-1, vmax=1, square=True,
                    center=0, annot=True, annot_kws={"size": 8}, cmap="PRGn")
    # plt.show()
    return plt

def split_chart(f_data, f_feature):
    ax = f_data[f_feature].value_counts(normalize=True).plot(kind='bar')
    return plt


def run_model(X_training, y_training, X_testing, y_testing, selected_model, selected_folds, selected_epochs):
    # ANN initialization
    with st.spinner("Building and evaluating model..."):
        my_bar = st.progress(0)

        classifier = Sequential()

        if selected_model == '24-12-1':
            classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))
            classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
            classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

        elif selected_model == '24-24-12-1':
            classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))
            classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
            classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
            classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

        # ANN compilation
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        cross_val_round = 1

        my_bar.progress(1)

        for train_index, val_index in KFold(selected_folds, shuffle=True, random_state=selected_folds).split(X_training):
            x_train, x_val = X_training[train_index], X_training[val_index]
            y_train ,y_val = y_training[train_index], y_training[val_index]
            classifier.fit(x_train, y_train, epochs=selected_epochs, verbose=0)
            classifier_loss, classifier_accuracy = classifier.evaluate(x_val, y_val)
            st.write(f'Round {cross_val_round} - Loss: {classifier_loss:.4f} | Accuracy: {classifier_accuracy * 100:.2f} %')
            my_bar.progress(cross_val_round*(1/selected_folds))
            cross_val_round += 1


        y_pred = classifier.predict(X_testing)
        y_pred[y_pred <= 0.5] = 0
        y_pred[y_pred > 0.5] = 1

        cm = pd.DataFrame(data=confusion_matrix(y_testing, y_pred, labels=[0, 1]),
                          index=["Actual Unstable", "Actual Stable"],
                          columns=["Predicted Unstable", "Predicted Stable"])

    name_string = str(selected_model) + '_' + str(selected_folds) + '_' + str(selected_epochs)
    csv_string = name_string+".csv"
    model_string = name_string+".h5"

    cm.to_csv(csv_string)

    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    classifier.save(model_string)


    st.success('Finished building and evaluating model!')
    st.balloons()

    view_results(name_string, y_testing, selected_model, selected_folds, selected_epochs)

def view_results(name_string, y_testing, selected_model, selected_folds, selected_epochs):
    # model_string = name_string+".h5"
    # # load model
    # model = load_model(model_string)
    # # summarize model.
    # model.summary()

    st.subheader('Selected Model Results')
    st.write('Model Architecture: ' + str(selected_model))
    st.write('Fold Count: ' + str(selected_folds))
    st.write('Epoch Count: ' + str(selected_epochs))

    csv_string = name_string+".csv"
    cm = pd.read_csv(csv_string,index_col=0)
    st.write(cm)
    accuracy = (cm.iloc[0, 0] + cm.iloc[1, 1]) / len(y_testing) * 100
    st.write('Accuracy per the confusion matrix:')
    st.write(accuracy)


st.header('Exploratory Data Analysis')

st.subheader('Unstable vs Stable Split')

st.write(f'Split of "unstable" (0) and "stable" (1) observations in the original dataset:')
st.write((data['stabf'].value_counts(normalize=True)))

st.pyplot(split_chart(data,'stabf'))

st.subheader('Correlation Map')
st.write("It is important to verify the correlation between each numerical feature and the dependent variable, as well as correlation among numerical features leading to potential undesired colinearity. The heatmap below provides an overview of correlation between the dependent variable ('stabf') and the 12 numerical features. Note that also the alternative dependent variable ('stab') has been included just to give an idea of how correlated it is with 'stabf'. Such correlation is significant (-0.83), as it should be, which reinforces the decision to drop it, anticipated in Section 3. Also, correlation between 'p1' and its components 'p2', 'p3' and 'p4' is above average, as expected, but not as high to justify any removal.")
st.pyplot(correlation_map(data, 'stabf', 14))

st.subheader('Feature Exploration')

st.write("Distribution patterns and the relationship with the 'stab' dependent variable is charted for each of the 12 dataset features. As this data comes from simulations with predetermined fixed ranges for all features, as described in Section 3, distributions are pretty much uniform across the board, with the exception of 'p1' (absolute sum of 'p2', 'p3' and 'p4'), which follows a normal distribution (as expected) with a very small skew factor of -0.013.")
selected_feature = st.radio(
     "Select dataset feature to view distribution chart and relationship with the 'stab' dependent variable.",
     data.columns)


assessment(data, 'stab', selected_feature, -1)

#how data was split into test in train

def splittesttrain(data):
    X = data.iloc[:, :12]
    y = data.iloc[:, 13]

    X_training = X.iloc[:54000, :]
    y_training = y.iloc[:54000]

    X_testing = X.iloc[54000:, :]
    y_testing = y.iloc[54000:]

    X_training.to_csv('X_training.csv')
    y_training.to_csv('y_training.csv')
    X_testing.to_csv('X_testing.csv')
    y_testing.to_csv('y_testing.csv')

ratio_training = y_training['stabf'].value_counts(normalize=True)
ratio_testing = y_testing['stabf'].value_counts(normalize=True)

st.header('Building and Training Deep Learning Model')

st.subheader('Splitting Data into Test/Train and Feature Scaling')
st.write("As anticipated, the features dataset will contain all 12 original predictive features, while the label dataset will contain only 'stabf' ('stab' is dropped here).")
st.write("In addition, as the dataset has already been shuffled, the training set will receive the first 54,000 observations, while the testing set will accommodate the last 6,000.")
st.write("Even considering that the dataset is large enough and well behaved, the percentage of 'stable' and 'unstable' observations is computed for both training and testing sets, just to make sure that the original dataset distribution is maintained after the split - which proved to be the case.")
st.write("After splitting, Pandas dataframes and series are transformed into Numpy arrays for the remainder of the exercise.")

splitcol1, splitcol2 = st.columns(2)

with splitcol1:
    st.write('Composition of Train Data')
    st.write(ratio_training)

with splitcol2:
    st.write('Composition of Test Data')
    st.write(ratio_testing)

st.write("In preparation for machine learning, scaling is performed based on (fitted to) the training set and applied (with the 'transform' method) to both training and testing sets.")

X_training = X_training.values
y_training = y_training.values

X_testing = X_testing.values
y_testing = y_testing.values

scaler = StandardScaler()
X_training = scaler.fit_transform(X_training)
X_testing = scaler.transform(X_testing)


st.subheader('Configure Deep Learning Model')

selected_model = st.radio("Select model architecture.",['24-12-1', '24-24-12-1'])
selected_folds = st.slider("Select number of folds.", min_value=5, max_value=10, value=5, step=5)
selected_epochs = st.slider("Select number of epochs.", min_value=10, max_value=50, value=10, step=20)


try:
    name_string = str(selected_model) + '_' + str(selected_folds) + '_' + str(selected_epochs)
    csv_string = name_string+'.csv'
    cm = pd.read_csv(csv_string,index_col=0)
    st.write('Model already trained with the selected parameters. Results displayed below.')
    view_results(name_string, y_testing, selected_model, selected_folds, selected_epochs)
except:
  col1, col2, col3 , col4, col5 = st.columns(5)

  with col1:
      pass
  with col2:
      pass
  with col4:
      pass
  with col5:
      pass
  with col3 :
      submit = st.button('Train model ⚡')

  if submit:
      run_model(X_training, y_training, X_testing, y_testing, selected_model, selected_folds, selected_epochs)
  else:
      st.write('Currently, there is no model trained with the selected parameters.')

st.header('Final Results')

labels = ['5 f, 10 e', '5 f, 30 e', '5 f, 50 e', '10 f, 10 e', '10 f, 30 e', '10 f, 50 e']
men_means = [95.66666666666667, 97.11666666666666, 97.25, 96.91666666666666, 97.21666666666667, 97.36666666666667]
women_means = [97.01666666666667, 97.45, 97.83333333333334, 97.91666666666666, 98.08333333333333, 97.88333333333334]

x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='24-12-1')
rects2 = ax.bar(x + width/2, women_means, width, label='24-24-12-1')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy Scores (%)')
ax.set_title('Accuracy Scores by Model Architecture and Parameters')
ax.set_xticks(x, labels)
ax.legend(loc='lower right')

ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=6)
ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=6)
ax.set_ylim(top=110)

fig.tight_layout()

st.pyplot(plt)
