
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler as mms, StandardScaler
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import ipywidgets as widgets
from IPython.display import display
from sklearn.svm import SVC
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
#solutions



#(LABEL ENCODING STRING FIELDS)
################################################################################

#create count chart for each string field
#note this down for each field
#if the field has a reasonable number of unique data use numberical labelling
#figure out what to do if the field has a unreasonable number of unique data

################################################################################

#what am I looking for with my graphs & what do I use that to impliment.

#looking for dominating classes so that i can give a balanced dataset to the model

#outlier identification & removal (how do I do that when the dataset is so large I cannot identify individual points)

#I want to be able to find out for example how many category 3 collisions happened in 2020

#count frequency of NA or missing data & decide if its better to drop or change to mean for that data. Issue: surely this will depend on the target variable, but then I can change the preprocessing based on the target vairiable
#change preprocessing based on target variable
#forward/backward fill for organised data
#or based on the target variable we are trying to predict (mean where target variable class == target variable class of missing data)


#label encoding (one hot encoding): using an individual field to state wether each data entry is in a specific class using binary 0 or 1. Means you can have a data entry be multiple classes

#do label encoding instead of creating a separate table of string type data

#make sure data is always correctly normalized and make sure the target variable is not normalized

#robust scaler to deal with outliers, choose between minmaxscaler & robust scaler and other scalers based on information from the target variable (number of outliers, missing data etc)

#one vs all & one vs one methods of doing multiple classification,

#  one vs all  sets one class vs all other classes.  (n)
# So for example if you had orange, apple & lemon it may create a class for apple & non apple where non apple is both orange & lemon and repeat until I know which class
# you will need up to one classifier per class, so for this example you would need to run up to 3 classifiers

#in one vs one it will pick only two classes & pit them against each other and see which works better (n *(n-1)/2)

#one vs 1 works better for 3 classes
#one vs all works better for 4 <= classes

#####################################

#my entire issue with trying to plot and understand the data is due to the massive dimensionality especially when for example pairplotting the entire dataset.
#I can reduce this dimensionality using PCA, so using PCA try to combine collumns and reduce the dimensionality of the dataset

#this involves feature selection (mincorr & correlation graphs)

#feature selection also involves trying to figure out which collumns contain similar information where it is redundant to have both for a specific analysis
#this type of reducing the dimensionality of collumns with SIMILAR information is PCA

#in the case of feature selection we are dropping useless features
#in the case of dimensionality reduction we are combining similar features to reduce redundancy

#question, should I never drop features and instead preform dimensionality reduction, or should i still drop uncorrelated features
####################################
colls = pd.read_csv("C:\\Users\\axeus\\Downloads\\Filtered_Sheffield_Traffic_Data.csv")

EDAsample = colls.sample(frac=0.1, random_state=42)
colls.isna().sum()
#could use unique on collumns with a limited choice of values to set them as categories. For example if the data type is 'string' I could have the values be unique and then set those as classes somehow

#fill the NA values with the mean of the respective columns.
#drop all string collumns 

numcolls = colls.select_dtypes(include=['float64', 'int64'])
stringcolls = colls.select_dtypes(exclude=['float64', 'int64']) #switch to label encoding here
#scale the data using min-max scaler
def preprocessing(colls, targetvariable):
       pass
def gethistplot(colls):
       fig,ax = plt.subplots(figsize=(10,10))
       sns.histplot(colls['local_authority_highway'])
#input mean for all NA data
numcolls.fillna(numcolls.mean(), inplace=True)
#change this to looping through the columns & if its a numerical value replacing with the mean, if its a string value replacing with the mode 
#with the exception of indexes (e.g collision_severity) which can be removed

#use relationship graph to find a list of the most correlated data to find the accident severity
corr_matrix = numcolls.corr()

#print(corr_matrix['collision_severity']['local_authority_district'] < 0.05)x
#number_of_vehicles
def getboxplot(colls,y,hue):
       sns.boxplot(
       data = colls,
       y='number_of_vehicles',
       palette='light:skyblue',
       linewidth=2,
       fliersize=8,
       hue='collision_severity',
       flierprops=dict(marker='o', markerfacecolor='red', markersize=8, alpha=0.7),
       medianprops=dict(color='orange', linewidth=2)
       )
       plt.show()
       print("BOX")

#getboxplot(colls)

def getviolinplot(colls,y,hue):
       sns.violinplot(
       data=colls,
       y='number_of_vehicles',
       color='skyblue',
       inner='box',
       hue='collision_severity',
       palette='Set2',
       linewidth=2,
       density_norm='width'
       )
       plt.show()
#getviolinplot(colls)

def getscatterplot(colls,x,y):
       sns.scatterplot(
       data=colls,
       hue = 'collision_severity',
       x=x,
       y=y,

       palette='Set1')
       plt.show()

#getscatterplot(colls,'collision_year','collision_severity')

#collision_adjusted_severity_slight(0.91)
#collision_adjusted_severity_serious(-0.59)
#enhanced_severity_collision (-0.14)
#number_of_casualties (-0.11)
#light_conditions (-0.9)
#number_of_vehicles(0.09)
#speed_limit(-0.07)
#truck_road_flag (0.07)
#urban_or_rural_area (0.6)
#collision_year (0.05)
def getpairplot(colls,xvals,yvals,hue):
       sns.pairplot(colls,x_vars =xvals,y_vars = yvals,hue=hue)
       print("pairplot")
       plt.show()
#getpairplot(colls,['collision year'],['urban_or_rural_area','number_of_casualties','light_conditions','number_of_vehicles','speed_limit'],'collision_severity')

def getcountplot(colls,xclass,classlabel):
       sns.countplot(data = colls, x = xclass)
       
       plt.xlabel(classlabel)
       plt.ylabel('count')
       plt.show()
       print("countplot")
#getcountplot(colls,'collision_severity')
# Source - https://stackoverflow.com/q/27241253
# Posted by yoshiserry, modified by community. See post 'Timeline' for change history
# Retrieved 2026-06-25, License - CC BY-SA 3.0

column_list = stringcolls.columns.values.tolist()
for column_name in column_list:
      print(stringcolls[column_name].unique())
      print("\n", column_name)
for column_name in column_list:
      print(stringcolls[column_name].unique().size)
      print("\n", column_name)

#getcountplot(stringcolls,stringcolls.columns.values[0],stringcolls.columns.values[0])
#for field in stringcolls:
#       print("field = " ,field)
#       getcountplot(stringcolls,field)

x = []
y = 'collision_severity'
def findcorrx(numcolls,corr_matrix,mincorr,y_name):
       x_name = []
       for column in numcolls.items():
              if corr_matrix[y_name][column[0]] < -mincorr or corr_matrix[y_name][column[0]] > mincorr and column[0] != y_name:
                     x_name.append(column[0])
       x = numcolls[x_name]
       return x, x_name

def SVM(colls, corr_matrix, mincorr, K, y_name,C,kernel,class_weight):
       #support vector classification
       
       y = colls[y_name]#for collision severity there are 3 classes to draw a decision boundary for. I should try to impliment one vs one classification. 
       #I could use a function to pre-process data when I am getting the correlations to apply a one vs one format or one vs many format
       x, x_name = findcorrx(numcolls,corr_matrix,mincorr,y_name)
       scaler = mms()
       x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
       
       model = SVC(C=C, kernel=kernel, class_weight=class_weight)
       x_train_scaled = scaler.fit_transform(x_train)
       x_test_scaled = scaler.transform(x_test)
       model.fit(x_train_scaled, y_train)
       score = model.score(x_test_scaled, y_test)
       return model,score,x_name

def findbestSVM(colls, numcolls, K, mincorr, y_name, slow):
       #find the best SVM model by testing different kernels and C values
       X = numcolls.drop(columns=[y_name])
       y = colls[y_name]
       pca = PCA(n_components=2)
    #find the polynomial relationship between the features & result 

    #methods (should apply as many as possible):
    #use testing to find the first polynomical with a good accuracy or use the same spread method as earlier
       if slow:
              X = pca.fit_transform(X)
              X_train,features_temporary,y_train,label_temporary = train_test_split(X,y,train_size=.70,random_state=10)
              X_valid,X_test,y_valid,y_test = train_test_split(features_temporary,label_temporary,train_size=.50,random_state=10)
              scaler = mms()
              X_scaled_train = scaler.fit_transform(X_train)
              X_scaled_valid = scaler.transform(X_valid)
              X_scaled_test = scaler.transform(X_test)
              param_grid = {
              "C": [0.1, 1, 10],
              "gamma": ["scale", 0.1, 0.01],
              "kernel": ["linear", "rbf", "poly"],
              "degree": [2, 3, 4]}
              svm = SVC(class_weight="balanced", probability=True)
              from sklearn.model_selection import GridSearchCV
              grid_search = GridSearchCV(
              estimator=svm,
              param_grid=param_grid,
              cv=5,
              scoring="accuracy",
              verbose=1,
              return_train_score=True
              )
              grid_search.fit(X_scaled_train, y_train)
              results_df = pd.DataFrame(grid_search.cv_results_)
              best_model = grid_search.best_estimator_
              X_train_valid = np.vstack((X_scaled_train, X_scaled_valid))
              y_train_valid = np.hstack((y_train, y_valid))
              best_model.fit(X_train_valid, y_train_valid)
              y_pred = best_model.predict(X_scaled_test)
              print("Best Hyperparameters:", grid_search.best_params_)
              print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
       else:
              kernels = ['linear', 'rbf', 'poly']
              C_values = [0.1, 1, 10]
              best_score = 0
              best_kernel = None
              best_C = None
              bestmodel = None
              modeldict = {}
              for kernel in kernels:
                     for C in C_values:
                            model,score,x_name = SVM(colls, corr_matrix, mincorr, K, y_name,C,kernel,'balanced')
                            modeldict[model] = score
                            if bestmodel != None:
                                   if score > modeldict[bestmodel]:
                                          best_score = score
                                          best_kernel = kernel
                                          best_C = C
                                          bestmodel = model
                            else:
                                   best_score = score
                                   best_kernel = kernel
                                   best_C = C
                                   bestmodel = model
              print("best kernel: " + best_kernel)
              print("best C value: " + str(best_C))
              print("best score: " + str(best_score))
              return bestmodel, x_name,best_score

       

def supervised_model(colls,corr_matrix,mincorr,K,y_name):
       #find optimum k-value /
       #find optimum mincorr value /
       #enter data & recieve y result /
       #
       #for finding optimum K value have a current K value, then find a range of values either side of the current K value and set K to the lowest among them and repeat until K does not change
       y = colls[y_name]
       x, x_name = findcorrx(numcolls,corr_matrix,mincorr,y_name)
       x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
       scaler = mms()
       severity_model = knn(n_neighbors=K)
       #increasing accuracy by removing the impact of outliers by scaling the training data
       x_train_scaled = scaler.fit_transform(x_train)
       x_test_scaled = scaler.transform(x_test)
       severity_model.fit(x_train_scaled, y_train)
       print(severity_model.score(x_test_scaled, y_test))
       return severity_model, x, x_name,y_name, severity_model.score(x_test_scaled, y_test)

def findmincorr(colls,corr_matrix,K,degree,mincorr,y_name,corrRange):#modeltype
       numcolls = colls.select_dtypes(include=['float64', 'int64'])
       if K != None:
              model,x,x_name,y_name,accuracy = supervised_model(colls,corr_matrix,mincorr,K,y_name)
       else:
              
              model, accuracy = polyregression(colls,corr_matrix,mincorr, numcolls, y_name,degree)
       mincorrdict = {mincorr:accuracy}
       
       while True:
              
              pointer = mincorr
              for i in np.arange(0.01,corrRange,0.01):
                     if corr_matrix[y_name].drop(y_name).max() < mincorr+i:

                            break
                     if K != None:
                            model, x, x_name, y_name, accuracy = supervised_model(colls,corr_matrix,mincorr+i,K,y_name)
                     else:
                            model, accuracy = polyregression(colls,corr_matrix,mincorr, numcolls, y_name,degree)
                     mincorrdict[mincorr+i] = accuracy
              for i in np.arange(0.1,corrRange,0.01):
                     if corr_matrix[y_name].drop(y_name).min() > mincorr-i or mincorr-i < 0:

                            break
                     if K != None:
                            model, x, x_name, y_name, accuracy = supervised_model(colls,corr_matrix,mincorr-i,K,y_name)
                     else:
                            model, accuracy = polyregression(colls,corr_matrix,mincorr, numcolls, y_name,degree)
                     mincorrdict[mincorr-i] = accuracy
              mincorr = max(mincorrdict, key=mincorrdict.get)
              if mincorr == pointer:
                     print("optimum mincorr value found: " + str(mincorr))
                     break
       return mincorr

def findK(colls,corr_matrix,mincorr,K,y_name,KRange,corrRange, slow):
       #for finding optimum K value have a current K value, then find a range of values either side of the current K value and set K to the lowest among them and repeat until K does not change
       model,x,x_name,y_name,accuracy = supervised_model(colls,corr_matrix,mincorr,K,y_name)
       Kdict = {K:accuracy}
       while True:
              
              pointer = K
              for i in range(1,KRange):
                     if slow == True:
                            mincorr = findmincorr(colls,corr_matrix,K,mincorr,y_name,corrRange)
                     model, x, x_name, y_name, accuracy = supervised_model(colls,corr_matrix,mincorr,K+i,y_name)
                     Kdict[K+i] = accuracy
              for i in range(1,KRange):
                     if K-i <1:
                            break
                     if slow == True:
                            mincorr = findmincorr(colls,corr_matrix,K,None,mincorr,y_name,corrRange)
                     model, x, x_name, y_name, accuracy = supervised_model(colls,corr_matrix,mincorr,K-i,y_name)
                     Kdict[K-i] = accuracy
              K = max(Kdict, key=Kdict.get)
              #exponential
  
              print(Kdict)
              print(K)
              if K == pointer:
                     print("optimum K value found: " + str(K))
                     break
              accuracy = Kdict[K]
       if slow == False:
              #colls,corr_matrix,K,degree,mincorr,y_name,corrRange
              mincorr = findmincorr(colls,corr_matrix,K,None,mincorr,y_name,corrRange)
       return model,x,x_name,y_name,accuracy,K,mincorr
#slow version finding mincorr within k
#fast version finding k first then mincorr



def slider (model, x,y):
       #create a slider for each collumn in x
       sliders = []
       for column in x.columns:
              slider = widgets.FloatSlider(value=0, min=x[column].min(), max=x[column].max(), step=0.1, description=column)
              sliders.append(slider)
       display(*sliders)

def predict(model,x_name,colls):
       x_values = []
       #use sliders for input
       for feature in x_name:
              value = input(f"Enter the value for {feature}: ")
              if value == "nan":
                     value = colls[feature].mean()
              else:
                     value = float(value)
              x_values.append(value)

       print(model.predict(np.array([x_values])))



#slider(model, x,y)
def corrmap(corr_matrix):
    plt.figure(figsize=(25,20))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.show()

#in order:
#collision_adjusted_severity_slight(0.91)
#collision_adjusted_severity_serious(-0.59)
#enhanced_severity_collision (-0.14)
#number_of_casualties (-0.11)
#light_conditions (-0.9)
#number_of_vehicles(0.09)
#speed_limit(-0.07)
#truck_road_flag (0.07)
#urban_or_rural_area (0.6)
#collision_year (0.05)

#accuracy could be increased by not using severity values if originally n/a and using them if they aren't n/a
#could do this by only changing NAN --> mean in the training data
#does not work as the data still has NAN values

#accuracy could be increased by finding the optimal num_neighbors using a graph visualisation

#accuracy could be increased by testing different value sets with different correlations

def linregression(colls,corr_matrix,mincorr, numcolls, y_name): #takes the scaled versions of the features (x) correlated with the result (y)
       X, x_name = findcorrx(numcolls,corr_matrix,mincorr,y_name)
       y = colls[y_name]
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
       model = LinearRegression()
       scaler = mms()
       X_train_scaled = scaler.fit_transform(X_train)
       model.fit(X_train_scaled, y_train)
       predictions = model.predict(X_test)
       mse = metrics.mean_squared_error(y_test, predictions)
       rmse = np.sqrt(mse)
       return model,rmse,y_name
def polyregression(colls,corr_matrix,mincorr, numcolls, y_name,degree):
       X, x_name = findcorrx(numcolls,corr_matrix,mincorr,y_name)
       X.fillna(X.mean(), inplace=True)
       print(x_name)
       print(X)
       print("is nan exist", X.isna().sum())
       y = colls[y_name]
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
       polyify = PolynomialFeatures(degree=degree)
       scaler = mms()
       X_train_scaled = scaler.fit_transform(X_train)
       X_train_poly = polyify.fit_transform(X_train_scaled)
       x_test_poly = polyify.transform(scaler.transform(X_test))
       model = LinearRegression()
       model.fit(X_train_poly, y_train)
       predictions = model.predict(x_test_poly)
       mse = metrics.mean_squared_error(y_test, predictions)
       rmse = np.sqrt(mse)
       return model,rmse



def getbestdegree(colls,corr_matrix,mincorr,numcolls,y_name,DRange,degree,corrRange,slow):
       #for finding optimum K value have a current K value, then find a range of values either side of the current K value and set K to the lowest among them and repeat until K does not change
       model,accuracy = polyregression(colls,corr_matrix,mincorr, numcolls, y_name,degree)
       print("current degree1: " + str(degree))
       print("current accuracy: " + str(accuracy))
       Ddict = {degree:accuracy}
       while True:
              print("loopies")
              print(DRange)
              print("current degree: " + str(degree))
              pointer = degree
              print("pointer: " + str(pointer))
              for i in range(1,DRange):
                     if slow == True:
                            mincorr = findmincorr(colls,corr_matrix,None,degree,mincorr,y_name,corrRange)
                     model, accuracy = polyregression(colls,corr_matrix,mincorr, numcolls, y_name,degree+i)
                     Ddict[degree+i] = accuracy
                     print("degree: " + str(degree+i))
                     print("accuracy: " + str(accuracy))
              for i in range(1,DRange):
                     if degree-i <1:
                            break
                     if slow == True:
                            mincorr = findmincorr(colls,corr_matrix,None,degree,mincorr,y_name,corrRange)
                     model, accuracy = polyregression(colls,corr_matrix,mincorr, numcolls, y_name,degree-i)
                     Ddict[degree-i] = accuracy
                     print("degree: " + str(degree-i))
                     print("accuracy: " + str(accuracy))
              degree = min(Ddict)
              #exponential
              print("Min = " + str(min(Ddict)))
              print(Ddict)
              print(degree)
              if degree == pointer:
                     print("optimum degree found: " + str(degree))
                     break
              accuracy = Ddict[degree]
       if slow == False:
              mincorr = findmincorr(colls,corr_matrix,None,degree,mincorr,y_name,corrRange)
       print(Ddict)
       return model,accuracy,y_name,degree
#

#enhanced_severity_collision (-0.14)
#number_of_casualties (-0.11)
#light_conditions (-0.9)
#number_of_vehicles(0.09)
#speed_limit(-0.07)
#truck_road_flag (0.07)
#urban_or_rural_area (0.6)
#collision_year (0.05)


#create a model that can automatically figure out the collumns with relevant correlation to the entered target data
#improve by finding the optimal value for the minimum necessary correlaton through testing
#automatically/manually find optimum k-value



def linregression_graph(y_test, predictions):
       plt.figure(figsize=(8, 6))

       # Scatter plot: Actual vs Predicted values
       plt.scatter(y_test, predictions, edgecolor='black', alpha=0.7, color='plum', label='Predicted Points')

       # Regression line (best fit) through predicted vs actual values
       z = np.polyfit(y_test, predictions, 1)  # Linear fit (degree=1)
       p = np.poly1d(z)
       plt.plot(y_test, p(y_test), color='red', linewidth=2, label='Regression Line')

       # Perfect prediction line (y=x)
       plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='green', linewidth=2, label='Perfect Prediction')

       # Labels and Title
       plt.xlabel('Actual Severity', fontsize=12, weight='bold')
       plt.ylabel('Predicted Severity', fontsize=12, weight='bold')
       plt.title('Actual vs Predicted Severity with Regression Line', fontsize=14, weight='bold')

       plt.legend()
       plt.grid(True, linestyle='--', alpha=0.4)
       plt.show()

def menu():
       choice = 1#int(input("enter the option you want:\n 1) create supervised model"))
       mincorr= 0.06
       K = 5
       slow = False
       #print(stringcolls.columns)

       y_name = input("Enter the name of the collumn you want to predict, options:\n" + str(numcolls.columns) + "\n")
       choice = int(input("if you are trying to predict a class based on input data enter 1 for neighbors model or 2 for SVM, if you are trying to predict a value based on input data enter 3 for linear regression or 4 for polynomial regression\n"))
       #if the user wants to enter a string y point then use a correlation model
       if choice == 1:
              #dataframe, correlation matrix, minimum correlation, K value, name of the collumn you want to predict, range for K value testing, range for mincorr testing, slow/fast
              model,x,x_name, y_name,accuracy,mincorr,K =findK(colls,corr_matrix,mincorr,K,y_name,5,5,slow)
              predict(model,x_name,colls)
       elif choice == 2:
              model,x_name, accuracy = findbestSVM(colls,numcolls,K,mincorr,y_name, slow)
       elif choice == 3:
              model, rmse,x_name = linregression(colls,corr_matrix,mincorr,numcolls,y_name)
       elif choice == 4:
              model,rmse, y_name,degree,x_name = getbestdegree(colls,corr_matrix,mincorr,numcolls,y_name,3,5,5,slow)
              print("accuracy: " + str(rmse))
              model.predict(np.array([numcolls[x_name].mean()]))



#make regression model where you enter two values, where one predicts the other.

#collision severity: 1,2,3 fatal, serious, slight
#label = collision_severity + collision_adjusted_severity_[serious&slight]

#collision_index
#collums with na data: location_easting&northing_osgr, latitude & longitude, local_authority_highway_current, collision_adjustest_severity_[serious&slight]

#lower is more severe

#list of contributing factors:
#collision_year (0.05)
#collision_adjusted_severity_slight(0.91)
#collision_adjusted_severity_serious(-0.59)
#enhanced_severity_collision (-0.14)
#truck_road_flag (0.07)
#urban_or_rural_area (0.6)
#light_conditions (-0.9)
#speed_limit(-0.07)
#num_casualties (-0.11)
#number_of_vehicles(0.09)

#in order:
#collision_adjusted_severity_slight(0.91)
#collision_adjusted_severity_serious(-0.59)
#enhanced_severity_collision (-0.14)
#num_casualties (-0.11)
#light_conditions (-0.9)
#number_of_vehicles(0.09)
#speed_limit(-0.07)
#truck_road_flag (0.07)
#urban_or_rural_area (0.6)
#collision_year (0.05)

####################################################################################################################################
#create a feature where you input as many valued details as possible and returns the likely severity of that collision

#need to calculate all severity scores
#need to classify severity as 1,2,3 (int) instead of float (1-3)

#KNOWLEDGE
#find out how to enter a list of data and get a severity result
#find out how to get multiple result fields
#find out how to do unsupervised learning
#find k-fold
#find hyperparameter
#find cross-validation


#FIXES/FEATURES
#create simple frontend? text based menu first
#make the model only give integer values for severity (correct data types)

#NEIGHBORS SUPERVISED MODEL
#choose the y & list of x, then predict y based on x all of which can be any data
#first get severity based on all numerical data
#add str data to the model
#find optimal neighbors value

#REGRESSION MODEL
#make a regression model
#use PCA to create the regression model
#try to find polynomial relationships using graphs & create a single & multiple regression model
#create a function which creates a regression model that can work on 1+ data collumns

#UNSUPERVISED MODEL
#clustering?


#EXTRA/UNIVERSAL
#identify possible classification problems
#identify all meaningful clustering tasks
#apply validation for clustering
#accuracy metrics (accuracy, precision, recall, F1-score, confusion matrix, Mean Absolute Error, MeanSquared Error, Root Mean Squared Error, Silhouette Score, Elbow Method)
#font end

#FUNCTIONS

#Graphs & data input to see the effect of different features on the severity of the collision (essentially an easy way to view the correlation matrix of the features & figure out danger of the road) - unsupervised model

#find the opimum values for reducing severity/your chosen target data (Linear regression model)


#categorise multiclass, binar & categorical data (supervised model) essentially more purposeful and easily readable results. multiclass = case, binary = true/false, categorical = choose from list of varied categories
#need to look properly into all possible result data for this and do extensive testing likely including graphs.
#I think ignore until validation stage. Currently the unsupervised model is complete and only needs the accuracy increasing

#essentially filling in missing data

#use regression to try and predict an entered data point. Have sliders to change the other affecting data points or data entry and then display the likely result
#essentially finding a value from limited data 