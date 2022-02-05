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

st.set_page_config(page_title="predicting grid stability", page_icon=":zap:", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)

st.sidebar.title('Predicting Grid Stability with Deep Learning')
st.sidebar.image("img/logo.png", use_column_width='auto')
st.sidebar.write("In this exercise, I have adapted Paolo's work into this streamlit app to allow for more model interactivity and testing.")
st.sidebar.write("Source: [Paolo Breviglieri](https://www.kaggle.com/pcbreviglieri/predicting-smart-grid-stability-with-deep-learning/notebook)")

# st.sidebar.header('Problem Statement')
# st.sidebar.write('(explain what the project is about)')

DATE_COLUMN = 'date/time'
DATA_URL = ('smart_grid_stability_augmented.csv')

@st.cache
def load_data():
    sns.set()
    start_time = datetime.now()
    data = pd.read_csv(DATA_URL)

    map1 = {'unstable': 0, 'stable': 1}
    data['stabf'] = data['stabf'].replace(map1)
    data = data.sample(frac=1)

    X_training = pd.read_csv('X_training.csv', index_col=0)
    y_training = pd.read_csv('y_training.csv', index_col=0)
    X_testing = pd.read_csv('X_testing.csv', index_col=0)
    y_testing = pd.read_csv('y_testing.csv', index_col=0)

    return data, X_training, y_training, X_testing, y_testing

st.header('Background')
st.subheader('Integrating Renewable Energy into the Electric Grid')

st.write("As renewable energy sources are increasingly adopted, an infrastructure of new paradigms is necessary taking into account more production sources and management of a much more flexible and complex system to connect producers, conumers, and distribution.")
st.write("Traditional operating ecosystems previously involved fewer energy sources that supply energy to consumers over unidirectional flows. However, with more renewable options, the end users now can consume energy and also have the ability to produce and supply it, yielding the new classification of 'prosumers'. Now, the energy flow within distribution grids, or 'smart grids' is bidirectional.")
st.write("This increased flexibility due to renewable sources and 'prosumers' has led to the management of supply and demand to be far more complex and challenging, leading to many studies looking into methods of predicting and managing smart grid stability.")

st.subheader('Modelling Grid Stability')

st.write("In a smart grid, consumer demand information is evaluated against the current supply conditions and the resulting price information is sent back to customers for them to decide about usage. This is highly time-dependent, so dynamically estimating grid stability is essential to the process.")
st.write("Overall, it is important to understand and plan energy production and consumption disturbances and fluctuations introduced by system participants in a dynamic way. This would need to consider technical aspects and how participants respond to changes in the price of energy.")
st.write("One approach that researchers have developed is that of Decentral Smart Grid Control (DSGC) systems. This methodology is based on monitoring the frequency of the grid. 'Frequency' refers to the alternate current (AC) frequency, measured in Hertz (Hz). Typically, a standard AC frequency of 50 or 60 Hz is utilized in electric power generation-distribution systems.")
st.write("Electrical signal frequency is known to increase in times of excess generation and decreases in times of underproduction.  Based on this, measurements of grid frequency for each customer would provide the required information about the current network power balance, to price the energy and inform consumers.")
st.write("The DSGC differential equation-based mathematical model identifies grid instability for a reference 4-node star architecture, with one power source (a centralized generation node) supplying energy to three consumption nodes. The model considers inputs (features) related to:")

st.write("- the total power balance (nominal power produced or consumed at each grid node)")
st.write("- the response time of participants to adjust consumption and/or production in response to price changes (referred to as reaction time)")
st.write("- energy price elasticity")

col1, col2 = st.columns(2)

with col1:
    st.image("img/node-diagram.png", caption='4-Node Star Diagram showing generation node that serves as the energy source and three consumption nodes.', width='None', use_column_width='auto')
with col2:
    st.image("img/logo.png", caption='4-Node Star with producer and consumers represented.', width='None', use_column_width='auto')


st.header('Dataset')
data_load_state = st.text('Loading data...')
data, X_training, y_training, X_testing, y_testing = load_data()
data_load_state.text("Dataset successfully loaded!")

st.write("This dataset (downloaded from [Kaggle](https://www.kaggle.com/pcbreviglieri/predicting-smart-grid-stability-with-deep-learning/notebook)) is synthetic and is based on simulations of grid stability for a reference 4-node star network. The Kaggle dataset is augmented from the 'Electrical Grid Stability Simulated Dataset', created by Vadim Arzamasov (Karlsruher Institut für Technologie) and is [available on the UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Electrical+Grid+Stability+Simulated+Data).")

st.write("For this project, we have shuffled the rows and here's a fragment of the data which contains predictive features and dependent variables that are further outlined below:")

st.write(data.head())


st.subheader('Predictive features:')
st.write("- 'tau1', 'tau2', 'tau3', 'tau4': reaction time of each network participant, a real value within the range 0.5 to 10 ('tau1' is the supplier node while 'tau2' to 'tau4' are the consumer nodes)")
st.write("- 'p1', 'p2', 'p3', 'p4': nominal power produced (positive) or consumed (negative) by each network participant, a real value within the range -2.0 to -0.5 for consumers ('p2' to 'p4')")
st.write("- 'g1', 'g2', 'g3', 'g4': price elasticity coefficient (gamma) for each network participant, a real value within the range 0.05 to 1.00 ('g1' is the supplier node while 'g2' to 'g4' are the consumer nodes)")

st.subheader('Dependent variables:')

st.write("- 'stab': maximum real part of the characteristic differential equation root (the system is linearly unstable if positive and linearly stable if negative)")
st.write("- 'stabf': categorical label ('stable' or 'unstable')")

st.write("The total power consumed equals the total power generated, so p1 (supplier node) = - (p2 + p3 + p4)")

if st.checkbox('Show raw data'):
    st.subheader('Shuffled dataframe')
    st.write(data)

# function to display histogram for each dependent independent varibale pair from data frame
def assessment(f_data, f_y_feature, f_x_feature, f_index=-1):
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

    st.pyplot(plt)


# function for heatmap of correlation of dataframe features
def correlation_map(f_data, f_feature, f_number):

    f_most_correlated = f_data.corr().nlargest(f_number,f_feature)[f_feature].index
    f_correlation = f_data[f_most_correlated].corr()

    f_mask = np.zeros_like(f_correlation)
    f_mask[np.triu_indices_from(f_mask)] = True
    with sns.axes_style("white"):
        f_fig, f_ax = plt.subplots(figsize=(20, 10))
        sns.heatmap(f_correlation, mask=f_mask, vmin=-1, vmax=1, square=True,
                    center=0, annot=True, annot_kws={"size": 8}, cmap="PRGn")
    return plt

# function to plot the split of stable vs unstable in the dataset
def split_chart(f_data, f_feature):
    ax = f_data[f_feature].value_counts(normalize=True).plot(kind='bar')
    return plt

# function to run ML models from button, compiled and trained models are then saved to json and results to csv that can be accessed later :)
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

# function to view results saved off earlier from previous runs
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

st.write(f'To understand the dataset, we can check the split of "unstable" (0) and "stable" (1) observations:')
st.write((data['stabf'].value_counts(normalize=True)))

st.pyplot(split_chart(data,'stabf'))

st.subheader('Correlation Map')
st.write("The correlation heatmap provides an overview of correlation between the dependent variable ('stabf') and the 12 numerical features and is shown below:")
st.pyplot(correlation_map(data, 'stabf', 14))

st.subheader('Feature Exploration')

st.write("To further understand the features, we can plot the distribution patterns and the relationship with the 'stab' dependent variable for each of the 12 dataset features.")
st.write("Since this data comes from simulations with predetermined fixed ranges for all features, distributions are pretty much uniform. The only exception is 'p1', which is the  sum of 'p2', 'p3' and 'p4', and follows a normal distribution (with a small skew factor of -0.013)")
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
st.write("The features dataset will contain the 12 predictive features, and the label dataset will only containe the'stabf' classification labels. The training portion will contain the first (shuffled) 54,000 observations, while the testing portion will consist of the last (shuffled) 6,000.")
st.write("The percentage of 'stable' and 'unstable' observations is computed to show approximate equivalence to the original dataset distribution:")

splitcol1, splitcol2 = st.columns(2)

with splitcol1:
    st.write('Composition of Train Data')
    st.write(ratio_training)

with splitcol2:
    st.write('Composition of Test Data')
    st.write(ratio_testing)

st.write("Finally, in preparation for inputting to the machine learning model, scaling is performed based on the training set and applied with the 'transform' method to both thr training and testing sets.")

X_training = X_training.values
y_training = y_training.values

X_testing = X_testing.values
y_testing = y_testing.values

scaler = StandardScaler()
X_training = scaler.fit_transform(X_training)
X_testing = scaler.transform(X_testing)

st.subheader('Design of Deep Learning Model')

st.write("For this application, we have two artificial neural network (ANN) architectures depicted and evaluated with the following structure:")
st.write("- one input layer (12 input nodes)")
st.write("- two or three hidden layers (24-12 or 24-24-12 nodes)")
st.write("- one single-node output layer")

st.write("These architectures can be summarized as follows:")
st.write("- 24-12-1")
st.write("- 24-24-12-1")

st.write("'relu' was chosen as the activation function for hidden layers and 'sigmoid' was the activation function for the output layers. Model compilation was performed with with 'adam' as optimizer and 'binary_crossentropy' as the loss function. Fitting performance is assessed with the 'accuracy' metric.")

st.image("img/ann.png", caption='Diagram of 24-24-12-1 Artificial Neural Network (ANN) architecture', width='None', use_column_width='auto')


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

st.write("Overall, we can see that deep learning is valuable in predicting the stability of the simulated grid based on this exercise. For this case, we can see that the best performance based on accuracy was with the more complex model architecture, 10 folds, and 30 epochs.")
