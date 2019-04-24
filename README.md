# Predicting UK Accident Severity with Decision Tree
This project will predict the severity of an accident in a specified spot, given the road, weather and light conditions.  

## Team Members
* [Sebastian Aldi](https://github.com/sebastianaldi17)
* [Dave J](https://github.com/djoshua449)
* [Matius Ebenhaezer](https://github.com/Ebnhzr)

## Prerequisites
* [Python (Version used is 3.7.1)](https://www.python.org/downloads/)
* Pip (Should be included with the python install if it is added to PATH)
  * If pip is not yet installed, refer to [this link](https://www.makeuseof.com/tag/install-pip-for-python/) for steps on how to install pip.
* Pandas 0.24.2
  * Open your command line or terminal, and then type `pip install pandas`.
* Plotly 3.7.1
  * Open your command line or terminal, and then type `pip install plotly`
* sklearn
  * sklearn requires scikit-learn 0.20.3, scipy 0.13.3 or higher, and numpy 1.8.2 or higher. Install those three using pip first by typing `pip install numpy scipy scikit-learn sklearn` on the terminal or command line.
* [The dataset used for this project](https://www.kaggle.com/tsiaras/uk-road-safety-accidents-and-vehicles)
  * Extract both csv files and put them in the same folder as the python script (`app.py`)
* A mapbox account. Create an account [here](https://www.mapbox.com/) and generate an access token. Your acces token can be found on the website like below:
![Mapbox account](https://raw.githubusercontent.com/sebastianaldi17/PredictAccidentSeverity/master/images/mapboxaccount.PNG)

## Opening the appllication
1. Open up `app.py` on any text editor, and replace some of the arguments:
* Replace `access_token` with your mapbox access token,
* Replace `sampling` to the amount of rows you want to load, higher will take the script to load longer.
* Replace `training_sample` to the amount of rows you want to use as training data, lower will cause lower accuracy, but higher can also cause lower accuracy due to overfitting.
2. If you have installed python, `app.py` can be double clicked immediately to launch the script. However, if double clicking fails, open the command line and navigate to the folder source (using `cd`) and then type `python app.py`. Launching the application should take quite some time (considering the dataset contains more than 2 million rows). Once the server is up, you should see something similar to the below picture in your command line or terminal:

![Command Prompt](https://raw.githubusercontent.com/sebastianaldi17/PredictAccidentSeverity/master/images/runapp.PNG)

3. Simply open `localhost:8050` or `127.0.0.1:8050` on any browser.

## Application
After performing step number 3, this page should show up on your browser:
![Landing page](https://raw.githubusercontent.com/sebastianaldi17/PredictAccidentSeverity/master/images/statistics.PNG)  
There are two tabs at the top: One displays the accuracy of the decision tree, as well as the confusion matrix.
![Predict page](https://raw.githubusercontent.com/sebastianaldi17/PredictAccidentSeverity/master/images/predicter.PNG)  
The other tab displays the map, as well as parameters that you can tweak to simulate conditions on a particular spot. The result will appear below the predict button.

## Code discussion
This section will explain the code, part by part.
### Part 1: Required libraries
![libraries](https://raw.githubusercontent.com/sebastianaldi17/PredictAccidentSeverity/master/images/requiredlibraries.PNG)
* Dash is the framework for the web application.
* Dash core components provide the medium for the graphs and maps, while Dash html components provides the html components, such as P, H1, Div, and others.
* Dash dependencies (Input, Output, State) provides a means to get the arguments from the components.
* Pandas is for reading csv files, as well as filtering and cleaning data.
* Plotly is used for the graph itself (dash core components simply provides a means for plotly to communicate with dash)
* DecisionTreeClassifier is the algorithm itself, so that we do not need to recreate it from scratch.
* LabelEncoder is used to convert strings into numbers so that Decision Tree can interpret the data.
* Confusion_Matrix is used to create a confusion matrix, which is an easy way to check the accuracy of the decision tree.
### Part 2: Reading, filtering, and cleaning data
![data cleaning](https://raw.githubusercontent.com/sebastianaldi17/PredictAccidentSeverity/master/images/datacleaning.PNG)
First we use pandas to load both csv files into a single dataframe, and then combine both of them for an easier time. Then, we select the columns that we want to use as features for the training data. In this case, the columns I want is the longitude, latitude, weather condition, road condition, and lighting condition. After that, we need to clean the data from unknown or missing values so that they do not interfere with the training data. Finally, we need to change the data from string into numbers that can be intepretted by the Decision Tree.

### Part 3: Creating the decision tree
![decision tree](https://raw.githubusercontent.com/sebastianaldi17/PredictAccidentSeverity/master/images/decisiontree.PNG)
From the data that we have cleaned, we take a random sample. The amount is set based on the `training_sample` argument, so that anyone can change how much data is sampled for the training. We separate the `Accident_Severity` column from the rest of the training data, and fit it into the decision tree. To check the accuracy of the decision tree, we need to create a confusion matrix (or count manually, but this way is faster), and take the amount of correct predictions divided by how many rows are there in the data (after cleaning).

### Part 4: The web application
We wil divide this section into some parts. The first is the base layout.
![layout](https://raw.githubusercontent.com/sebastianaldi17/PredictAccidentSeverity/master/images/framework.PNG)  
As you can see, the layout only contains a header for the title, and two tabs, as well as a div that will contain the content of the tabs. Changing tabs will be controlled by a function that is binded into the application callback.  
![tab1](https://raw.githubusercontent.com/sebastianaldi17/PredictAccidentSeverity/master/images/tab1.PNG)  
The output will be inside the div in the layout earlier, and the content of the tab itself is also a giant div, containing the layout for each tab. In the first tab, there is a heatmap which contains the confusion matrix (because plotly has no built in confusion matrix), as well as the text describing the accurate predictions and the accuracy of the decision tree.  
![tab2](https://raw.githubusercontent.com/sebastianaldi17/PredictAccidentSeverity/master/images/tab2.PNG)  
Second tab contains several drop-down menus for setting the weather, road and lighting conidtions, two sliders for longitude and latitude, and the map to display the current location and past accident locations from the data. The live updating from the sliders are also controlled through several callbacks, which I am going to describe below.
![liveupdate](https://raw.githubusercontent.com/sebastianaldi17/PredictAccidentSeverity/master/images/changeposition.PNG)  
The callback method above displays the creation of the map, as well as updating it upon dragging the longitude or latitude sliders. The map automatically repositions itself to the center of the longitude and latitude slider values, and also displays past accident locations. The amount of past accident markers cannot be too large, because in my experience chrome keeps crashing if I went past 20 thousand spots.  
![takeargs](https://raw.githubusercontent.com/sebastianaldi17/PredictAccidentSeverity/master/images/takeargs.PNG)  
This method takes all of the arguments from the sliders and drop-down menus, and then pass it to the decision tree so that it can predict how severe the accident will be. There needs to be a try-except (try-catch) block for the predicting because there is a chance that one of the drop-down menus is still empty/not selected. The output is then interpretted by the conditional branch. 2 means that the accident will be slight, 1 means that the accident will be serious, and 0 means that the accident will be fatal.
## References and acknowledgements
* Kaggle, for the dataset
* The developers of scikit-learn, for the decision tree algorithm, confusion matrix and label encoder
* [This article at Hackernoon, for helping me to decide which model to use](https://hackernoon.com/choosing-the-right-machine-learning-algorithm-68126944ce1f)
