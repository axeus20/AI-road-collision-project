
import pandas as pd
import scipy.stats
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
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from matplotlib.ticker import FormatStrFormatter
import random
from itertools import compress
from sklearn.model_selection import GridSearchCV
####################################################################################
#planning!

#create the basics of an SVM model:
# use the grid method to test a number of the various values (gamma,c,degree,kernel)
#use ovo & ova to create a models to increase classification accuracy which is essential for SVM and mildly helpful for knn
#return models and allow for basic input & prediction

#then the list of tastks is:

############
#increase validation. There are a number of methods that can be added to all of my current models to increase their accuracy & validation, these methods are:

# bias & variation (and trying to reduce both in models) using train error & difference in train error

# underfitting & overfitting (find code & practical examples of this and then attempt to impliment)

# grid search instead of my current method? this should be asked and can easily be implimented by removing the while loop for my method & adding a specific range

# cross validation (splitting up the dataset into a number of folds and finding the accuracy of each and the accuracy of every combination of the folds, then you can use this as your model)

# adding validation sets as an extra set fo data in train_test split --> train_test_validation split which should be around the same as the test (do not reduce training dataset size though) 

#adding more information about the accuracy of a model with performance metrics

#precision: using confusion metrics when the model gives a specific classification we can calculate the chance that this classification is correct
#so precision is how confident we are in a specific classified datapoint being correct instead of the overall accuracy of the model, 
# this can also help us understand bias & account for differences in accuracy for different classes

#a pretty good way to explain [accuracy vs precision vs recall] = 
# [overall chance of a correct prediction || chance that a positive result is actually positive || chance that a negative result is actually negative]

#f1 score is kind of an average between precision & recall
#############




#(LABEL ENCODING STRING FIELDS)
################################################################################

#create count chart for each string field
#note this down for each field
#if the field has a reasonable number of unique data use numberical labelling
#figure out what to do if the field has a unreasonable number of unique data
#labelencoder function detailed in 4.2

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


#the larger the variance in your data the more information you are given towards deciding a class/prediction from that feature

#I can see what features are very highly correlated and then use dimensionality reduction on them. For example collision_severity & collision_adjusted_severity

#also look into the variance of collumns to decide how valuable the data is as well as correlation, correlation + variance and then correlation between collumns

#for PCA use robust scaler to cut out outliers?

####################################
colls = pd.read_csv("C:\\Users\\alex\\Downloads\\Filtered_Sheffield_Traffic_Data.csv")

#EDAsample = colls.sample(frac=0.1, random_state=42)
colls.isna().sum()
#could use unique on collumns with a limited choice of values to set them as categories. For example if the data type is 'string' I could have the values be unique and then set those as classes somehow

#fill the NA values with the mean of the respective columns.
#drop all string collumns 

numcolls = colls.select_dtypes(include=['float64', 'int64'])
stringcolls = colls.select_dtypes(exclude=['float64', 'int64'])
## Source - https://stackoverflow.com/a/12850453
# Posted by Joran Beasley, modified by community. See post 'Timeline' for change history
# Retrieved 2026-06-27, License - CC BY-SA 4.0
#df_merged = pandas.concat([df1, df2], ignore_index=True, sort=False)

#scale the data using min-max scaler

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

#column_list = stringcolls.columns.values.tolist()
#for column_name in column_list:
#      print(stringcolls[column_name].unique())
#      print("\n", column_name)
#for column_name in column_list:
#      print(stringcolls[column_name].unique().size)
#      print("\n", column_name)

#getcountplot(stringcolls,stringcolls.columns.values[0],stringcolls.columns.values[0])
#for field in stringcolls:
#       print("field = " ,field)
#       getcountplot(stringcolls,field)
#returns a string list of names for the correlated columns 
#could create a separate numcolls & stringcolls processing as stringcolls processing does not always need to be done
  
def findcorrx(numcolls,corr_matrix,mincorr,y_name):
       x_name = []
       for column in numcolls.items():
              if corr_matrix[y_name][column[0]] < -mincorr or corr_matrix[y_name][column[0]] > mincorr and column[0] != y_name:
                     x_name.append(column[0])
       x = numcolls[x_name]
       return x, x_name
#dataframe, dataframe collumn name(string), boolean 

#preprocessing will generally be the first function called after menu and the start of any machine learning process.
#  It has different options for preprocessing based on the machine learning method.
def preprocessing(colls, targetvar,mincorr, init = True, pcamin=1,minvar = 0.8):
       #to find outliers compare each correlated feature of the target variable, use the IQR to find a number of outliers and scale the outlier % based on preformance testing. Do this BEFORE PCA
       #am I using robust or minmax scaler? --> check for outliers, if there are limited outliers use minmax, if there are many outliers use robust
       #which features can I use PCA on? --> check correlation between features and variation of features, if two features are highly correlated and one has little variance preform PCA, the acceptable variance can be based on preformance testing
       #which features should be dropped?
       #which for features should I fill in missing data or drop the entry
       #do I need to preform label encoding on the target variable? (classification vs prediction)
       #one hot or label encoding?

       #ordering: 
       # 1) label encoding, figure out if the target variable needs to be label encoded (if targetvar == string) /
       # 2) fill in missing data (mode/mean/drop) figure out how to choose what to drop (numcolls.fillna(numcolls.mean(), inplace=True)) /
       # 3) standardization (choose minmax or robust scaler & remove outliers)
       # 4) feature selection (get correlation to target variable & variance)
       # 5) PCA (get correlation between data & variance. If one feature has almost no variance then preform PCA, if both features are highly correlated then preform PCA use preformance testing to find this value)
       
       #for finding the mean it should surely be the mean of a specific class of the target variable, therefore I need to




       #label encode & find the mean of each column for each unique label/class
       #split into

       #here is preprocessing based on trying to classify data into a class by its features

       #colls = initial dataframe

       #strincorrs/numcorrs = split dataframe to preform preprocessing on string & numberical data separately

       #dfcorrs = final dataframe
       if init:

              #can you use string datatype in a knn classifier?
              colls[targetvar].dropna() #drops all rows where the target variable is null. These rows are not helpful.
              numcolls = colls.select_dtypes(include=['float64', 'int64'])
              stringcolls = colls.select_dtypes(exclude=['float64', 'int64'])
              #split into numcolls & stringcolls as I'm going based on the idea that non numerical features cannot help to classify data using a KNN model.
              #should I use labelling on the string features to transform them into numerical data that can then be used to help classify data?
              numcolls.fillna (numcolls.mean(), inplace=True)
              # Source - https://stackoverflow.com/a/51801256
              # Posted by Hans Musgrave
              # Retrieved 2026-07-01, License - CC BY-SA 4.0
              #print(numcolls['enhanced_severity_collision'].mean(), "MEAN = enhanced_severity_collision")
              #
              #print(numcolls['collision_injury_based'].mean() , "MEAN = collision_injury_based")

              #df.loc[df['col'] > 1990, 'First Season'] = 1
              for col in numcolls.columns:
                     #attempt at making it randomly replace -1s with -2,-1 or 0 for each value with a chance of -2, -1 or -0 being proportional to the mean,
                     #  meaning that the mean would barely change from before preprocessing to after instead of drastically changing for columns where -1 is over 50% of values

                     #if round(numcolls[col].mean()) == -1:
                     #       if numcolls[col].mean() < -1:
                     #              choicearr = [-2,-1]
                     #              choice = np.random.choice(choicearr,1,[(abs(numcolls[col].mean())-1), 1 - (abs(numcolls[col].mean())-1)])
                     #              numcolls.loc[numcolls[col] == -1 , col] = choice[0]
                     #       else:
                     #              choicearr = [-1,0]
                     #              choice = np.random.choice(choicearr,1,[(abs(numcolls[col].mean())), 1 - (abs(numcolls[col].mean()))])
                     #              print(choice)
                     #              numcolls.loc[numcolls[col] == -1 , col] = choice[0]
                     if round(numcolls[col].mean()) == -1 and numcolls[col].dtype == 'int64':
                            if numcolls[col].mean() > -1:
                                   numcolls.loc[numcolls[col] == -1 , col] = 0
                            else:
                                   numcolls.loc[numcolls[col] == -1 , col] = 1
                     else:
                            if numcolls[col].dtype == "int64":
                                   numcolls.loc[numcolls[col] == -1 , col] = round(numcolls[col].mean())
                            else:
                                   numcolls.loc[numcolls[col] == -1 , col] = numcolls[col].mean()
                     #print("numcolls[col].mean() = ", numcolls[col].mean())
                     #print("numcolls[cols].name = ", numcolls[col].name)
              for col in stringcolls.columns:
                     stringcolls.loc[stringcolls[col] == np.nan , col] = stringcolls[col].mode()[0]
                     if stringcolls[col].mode()[0] == -1:
                            stringcolls.loc[stringcolls[col] == -1 , col] = stringcolls[col].mode()[1]
                            #print("stringcolls[col].mode() = ", stringcolls[col].mode()[1])
                     else:
                            stringcolls.loc[stringcolls[col] == -1 , col] = stringcolls[col].mode()[0]
                            #print("stringcolls[col].mode() = ", stringcolls[col].mode()[0])
                     
                     
                     #print("stringcolls[cols].name = ", stringcolls[col].name)
              #the main issue I see with taking a complete mean is that if I am looking for example into collision severity 
              # if I take the mean of all class 1, 2 & 3 and place that into class 1,2 & 3 my dataset may be overwhelmed class 2 data 
              # for example even if the dataset is all class 1 & 3 but the mean falls into class 2 then I will fill all NAN datapoints with data suggesting class 2
              #however this may not be an issue as the model may weight this as 0 if its the mean since there is no variance.
              preprocessed_colls = pd.concat([numcolls, stringcolls], axis=1)
              labeler = LabelEncoder()
              if colls[targetvar].dtype != 'int64':
                     y = labeler.fit_transform(colls[targetvar])
              else:
                     y = colls[targetvar]
              #stringcolls.drop("local_authority_highway_current", axis=1, inplace=True) #without inplace=true
              #print(numcolls.isna().sum())
              #print(stringcolls.isna().sum())
              #split into numbers/non numbers
              corr_matrix = numcolls.corr()
              dfcorrs, corrnames = findcorrx(numcolls,corr_matrix,mincorr,targetvar)
              
              #remove outliers
              # Source - https://stackoverflow.com/a/23202269
              # Posted by tanemaki, modified by community. See post 'Timeline' for change history
              # Retrieved 2026-06-29, License - CC BY-SA 4.0
              #need to not remove outliers for features that cannot have outliers (classes or years)
              #I have chosen this for my outlier removal method as I could not figure out anything using graphs and I can't see an example using IQR to remove or detect outliers beyond robustscaler
              #I would need information to decide on robust or standardscaler and the only other way to do that is simply directly testing model accuracy
              #  instead of based on an outlier % which I think could be more accurate although it would use the same method of accuracy testing to improve
              #so the next question is, would it be better to simply test standardscaler vs robust scaler for my model as well vs remove those wih too manystandard deviations from the mean? 
              # especially since this would double the time taken, although O notation would be the same

              #change this to using IQR if possible
              removal_indexes = (np.abs(stats.zscore(dfcorrs)) > 4).all(axis=1) #creates an array where indexes 3 or more standard deviations from the mean are true and others are false
              #add some consideration for when outliers should be removed (when they are a reasonably insigificant portion of the dataset)
              #if they are too significant increase the boundary for outlier classification intil they are reasonably insignificant
              #I think a reasonable portion of outliers to remove would be around 2% of the dataset

              print("removal indexes = ", removal_indexes)
              #remcount = 0
              #for index,row in dfcorrs.iterrows():
              #       if removal_indexes[remcount]: #there is a value outside of tolerance in this row
              #              dfcorrs.drop(index)
              #       remcount+=1
              #massively increases the speed of removing outlier columns
              indexlist = list(compress(range(len(removal_indexes)), removal_indexes))
              dfcorrs.drop(dfcorrs.index[indexlist])


              #check for data outside IQR to choose robust or minmax scaler. IDK how to check for data quantity outside of iqr
              #choose standardisation 
              MMscaler = mms()
              Sscaler = StandardScaler()
              #print("sample5 = \n\n", numcolls.sample(20))

              #choosing between normalised or standardised data:

              #standardisation:
              #has a mean of 0 and standard deviation of 1
              #good for data with outliers as it does not use the min/max values to scale the data
              #good for features that don't follow the same scale

              #normalisation:
              #scales the data to a range of 0-1
              #good when working with models sensitive to magnitude of data (e.g. neural networks)
              
              #I'm unsure for the exact reason for using standardisation over normalisation here and mostly due to it being used for PCA in the examples from 4.1 & 4.2
              #the only thing I can think of is that many of the features do not follow a similar scale and therefore standardisation is more accurate than normalisation
              dfcorrs_scaled = Sscaler.fit_transform(dfcorrs)

                     #create an array of names for each column highly correlated to each other, ignoring variance
                     #then run the function on all combinations to see if all of those features can be combined, then remove a random one and repeat
              postpcanames=[]
              preprocessed_colls = []
              #count the number of columns to preform PCA on and then use that for n_components and let the PCA function automatically pick the specific columns
              #the main issue I can see with this method is I cannot specifically choose which features to combine and I cannot know if PCA will always pick the right ones
              pcacount = 0
              for cols1 in corrnames:
                     for cols2 in corrnames:
                            if cols2 == cols1 or cols1 in postpcanames or cols2 in postpcanames:
                                   continue
                            #print(cols1,"---", cols2)
                            #print(corr_matrix[cols1][cols2])
                            #the main issue with this pca check is that the function is very arbritrary, however I think it should work to simply scale it to the correct value for the target variable
                            #however it could still cause inacuracy issues & I will likely have to write another function to compare how valuable the reduced dimensionality is compared to marginally higher accuracy.
                            #essentially I'd need to know at what point do I stop improving the accuracy of the model and I can't be sure where that threshold is.
                            pcacheck1 = (abs(corr_matrix[cols1][cols2])-1) * numcolls[cols1].var()
                            pcacheck2 = (abs(corr_matrix[cols1][cols2])-1) * numcolls[cols2].var()
                            #print("pcacheck1 = ",pcacheck1)
                            #print("pcacheck2 = ", pcacheck2)
                            if (abs(corr_matrix[cols1][cols2])-1) * numcolls[cols1].var() < pcamin or (abs(corr_matrix[cols1][cols2])-1) * numcolls[cols2].var() < pcamin:
                                   pcacount +=1
                                   #this method is trying to combine specific features for PCA, which the second method is letting the PCA function do the work to choose which features combine
                                   #preform PCA
                                   #pca = PCA(n_components=2)
                                   postpcanames.append(cols1)
                                   postpcanames.append(cols2)
              pcacount = max(1,pcacount)
              #create a pca variance dictionary to find what value explains the most variance with the least number of dimensions.
              #options:
              #I could round variance to a certain degree (round to nearest 5 for example) and then make my tiebreaker the lower dimension value and then the actual better variance
              #I could use an arbitrary function with a weight for both degree & variance, e.g variance - degree
              #I could simply look for a non comprehensive value, only looking into a few n_components either side, proportional to the initial n_components, for example 20% either side with a min of 1
                                   #preprocessed_colls.append(pca.fit_transform(numcolls[[cols1, cols2]]))
              
              vardict = {}
              #1 = 3
              #4 = 3
              #8 = 5
              #12 = 7
              n_componentsarr = []*max(3,round(pcacount/2)+1)
              #making sure I'm looking through a small range around my initial

              #this doesn't work as it will always pick the highest value for PCA count, I need it to pick the the highest value within my minvar and if no values are in minvar then pick the highest
              #for i in np.arange(0,max(1,round(pcacount/4)),1):
              #       pca = PCA(n_components = pcacount+i)
              #       pca.fit(dfcorrs_scaled)
              #       
              #       explained_variance = pca.explained_variance_ratio_
              #       cumvar = np.cumsum(explained_variance)
              #       vardict[pcacount+i] = cumvar[-1]
              #       print("cumvar = ", cumvar)
              #       #n_componentsarr[list(vardict.keys()).index(pcacount+1)] = cumvar
              #       if pcacount - 1 < 1:
              #              continue
              #       pca = PCA(n_components = pcacount-i)
              #       pca.fit(dfcorrs_scaled)
              #       explained_variance = pca.explained_variance_ratio_
              #       cumvar = np.cumsum(explained_variance)
              #       print("cumvar = ", cumvar)
              #       vardict[pcacount+i] = cumvar[-1]
              #       #n_componentsarr[list(vardict.keys()).index(pcacount-1)] = cumvar
              #pcacount = max(vardict, key=vardict.get)
              
              #I did just realise I can do this by looking at the cumvar of the highest pcacount I'm willing to use and find the lowest pcacount that meets my explained variance standards
              maxpca = len(corrnames)
              pca = PCA(n_components = maxpca)
              pca.fit(dfcorrs_scaled)
              #I can find which dimension to stop at by checking the % increase of explained variance in the i+1 of variance ratio explained compared to cumvar of the same degree
              explained_variance = pca.explained_variance_ratio_
              cumvar = np.cumsum(explained_variance)

              #find the value where the increase in explained variance of i+1 is negligible compared to that of i

              #take the mean
              #if the value is significantly smaller than the mean then this is where you do not need to continue increasing the dimension
              #therefore if we have e.g 20 features that make up exactly 5% of the variance each, it will take every feature
              #however if we end up with something like 40,30,20,10 it will.. also take all features as the mean is 25 even though 10 can be discarded in this case
              #I could compare it to 1, but then we have the same problem of having to hard code the variance value
              #I think the only option is to try and calculate the trend using a comparission between % of previous and % of total
              #then if the % of previous is very low I can try to remove it, but if the % of total is high enough then I won't actually remove it.
              #I think a reasonable level for this threshold would be 0.15 since removing more than 15% of variance is probably not great
              #but this does assist our issue however if the data has a slowly decrementing variance it will not preform pca
              for i in range(len(cumvar)):
                     
                     if cumvar[i] > 0.9:
                            break
              pcacountvar = i

              #print("pcacount = ", pcacount)
              #print("postpcanames = ", postpcanames)
              pca = PCA(n_components=pcacountvar)


              preprocessed_colls = pd.concat([y,pd.DataFrame(pca.fit_transform(dfcorrs_scaled))], axis = 1)
       

       
       #here is preprocessing based on trying to predict a numberical value based on the data. I will impliment this after understanding the regression & prediction models
       else:
              print("NOT the initial preprocessing loop, this will be called for exclusively the re-do for preprocessing to change values mincorr & minvar")
       error_on_no_data = 1/len(pd.DataFrame(pca.fit_transform(dfcorrs_scaled)).columns.values)
       return y,pd.DataFrame(pca.fit_transform(dfcorrs_scaled)),corrnames,pca, Sscaler
       

#print(numcolls.sample(20))
#print(stringcolls.sample(20))

#dataframe, string(targetvar), float (mincorr), boolean (predict/classification)0
#preprocessed_colls = preprocessing(colls,'collision_severity',0.05,False)
#preprocessed_colls.describe()
  

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

       


#colls,corr_matrix,K,degree,mincorr,y_name,corrRange
#y_name = targetvar
#def findmincorr(colls,corr_matrix,mincorr,y_name,corrRange = 0.05):#modeltype
#       numcolls = colls.select_dtypes(include=['float64', 'int64'])
#       if K != None:
#              model,x,x_name,y_name,accuracy = supervised_model(colls,corr_matrix,mincorr,K,y_name)
#       else:
#              
#              model, accuracy = polyregression(colls,corr_matrix,mincorr, numcolls, y_name,degree)
#       mincorrdict = {mincorr:accuracy}
#       
#       while True:
#              
#              pointer = mincorr
#              for i in np.arange(0.01,corrRange,0.01):
#                     if corr_matrix[y_name].drop(y_name).max() < mincorr+i:

#                            break
#                     if K != None:
#                            model, x, x_name, y_name, accuracy = supervised_model(colls,corr_matrix,mincorr+i,K,y_name)
#                     else:
#                            model, accuracy = polyregression(colls,corr_matrix,mincorr, numcolls, y_name,degree)
#                     mincorrdict[mincorr+i] = accuracy
#              for i in np.arange(0.1,corrRange,0.01):
#                     if corr_matrix[y_name].drop(y_name).min() > mincorr-i or mincorr-i < 0:

#                            break
#                     if K != None:
#                            model, x, x_name, y_name, accuracy = supervised_model(colls,corr_matrix,mincorr-i,K,y_name)
#                     else:
#                            model, accuracy = polyregression(colls,corr_matrix,mincorr, numcolls, y_name,degree)
#                     mincorrdict[mincorr-i] = accuracy
#              mincorr = max(mincorrdict, key=mincorrdict.get)
#              if mincorr == pointer:
#                     print("optimum mincorr value found: " + str(mincorr))
#                     print("current model accuracy is = ", mincorrdict[mincorr])
#                     print("DICT = ", mincorrdict)
#                     break
#       return mincorr
#ZeroDivisionError

#def findK(colls,corr_matrix,mincorr,K,y_name,KRange,corrRange, slow):
#       #for finding optimum K value have a current K value, then find a range of values either side of the current K value and set K to the lowest among them and repeat until K does not change
#       model,x,x_name,y_name,accuracy = supervised_model(colls,corr_matrix,mincorr,K,y_name)
#       Kdict = {K:accuracy}
#       while True:
#              
#              pointer = K
#              for i in range(1,KRange):
#                     if slow == True:
#                            mincorr = findmincorr(colls,corr_matrix,K,mincorr,y_name,corrRange)
#                     model, x, x_name, y_name, accuracy = supervised_model(colls,corr_matrix,mincorr,K+i,y_name)
#                     Kdict[K+i] = accuracy
#             for i in range(1,KRange):
#                    if K-i <1:
#                           break
#                    if slow == True:
#                           mincorr = findmincorr(colls,corr_matrix,K,None,mincorr,y_name,corrRange)
#                    model, x, x_name, y_name, accuracy = supervised_model(colls,corr_matrix,mincorr,K-i,y_name)
#                    Kdict[K-i] = accuracy
#             K = max(Kdict, key=Kdict.get)
#             #exponential
# 
#             print(Kdict)
#             print(K)
#             if K == pointer:
#                    print("optimum K value found: " + str(K))
#                    break
#             accuracy = Kdict[K]
#      if slow == False:
#             #colls,corr_matrix,K,degree,mincorr,y_name,corrRange
#             mincorr = findmincorr(colls,corr_matrix,K,None,mincorr,y_name,corrRange)
#      return model,x,x_name,y_name,accuracy,K,mincorr
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



def linregression_graph(x_test,y_test, predictions):
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
def linregplots(y_test,predictions):
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
       plt.xlabel('Actual House Price (Y Test)', fontsize=12, weight='bold')
       plt.ylabel('Predicted House Price (Y Pred)', fontsize=12, weight='bold')
       plt.title('Actual vs Predicted House Prices with Regression Line', fontsize=14, weight='bold')

       plt.legend()
       plt.grid(True, linestyle='--', alpha=0.4)
       plt.show()
       print('MAE:', metrics.mean_absolute_error(y_test, predictions))
       print('MSE:', metrics.mean_squared_error(y_test, predictions))
       print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


       # --- Assume X_test, y_test, predictions exist ---
       X = x_test.to_numpy().reshape(-1)
       y = y_test.to_numpy().reshape(-1)

       # --- Normalize X and y for stable plotting ---
       X_mean, X_std = X.mean(), X.std()
       y_mean, y_std = y.mean(), y.std()
       X_norm = (X - X_mean) / X_std
       y_norm = (y - y_mean) / y_std

       # --- Assume predictions exist ---
       pred = predictions.reshape(-1)
       pred_norm = (pred - y_mean) / y_std

       # --- Create small grid around predictions for visualization ---
       # We'll simulate perturbing predictions slightly to see effect on MAE/MSE/RMSE
       delta = 0.1  # small variation
       theta0_vals = np.linspace(-delta, delta, 50)  # small intercept shift
       theta1_vals = np.linspace(0.9, 1.1, 50)      # small slope multiplier
       T0, T1 = np.meshgrid(theta0_vals, theta1_vals)

       # --- Initialize metric surfaces ---
       MAE_surface = np.zeros(T0.shape)
       MSE_surface = np.zeros(T0.shape)
       RMSE_surface = np.zeros(T0.shape)

       # --- Compute metrics for perturbed predictions ---
       for i in range(T0.shape[0]):
              for j in range(T0.shape[1]):
                     y_pred = T0[i, j] + T1[i, j] * pred_norm  # perturb predictions
                     MSE_surface[i, j] = metrics.mean_squared_error(y_norm, y_pred)
                     MAE_surface[i, j] = metrics.mean_absolute_error(y_norm, y_pred)
                     RMSE_surface[i, j] = np.sqrt(metrics.mean_squared_error(y_norm, y_pred))

       # --- Plot 3D surfaces ---
       fig = plt.figure(figsize=(22, 6))

       # ---- MAE subplot ----
       ax1 = fig.add_subplot(1, 3, 1, projection='3d')
       surf1 = ax1.plot_surface(T0, T1, MAE_surface, cmap='plasma', alpha=0.9, edgecolor='k', linewidth=0.2)
       ax1.set_xlabel("Theta0 Shift", labelpad=12)
       ax1.set_ylabel("Theta1 Multiplier", labelpad=12)
       ax1.set_zlabel("MAE", labelpad=15)
       ax1.set_title("MAE Surface", pad=20)
       ax1.view_init(elev=35, azim=135)
       ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
       ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
       ax1.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
       fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1).set_label('MAE')

       # ---- MSE subplot ----
       ax2 = fig.add_subplot(1, 3, 2, projection='3d')
       surf2 = ax2.plot_surface(T0, T1, MSE_surface, cmap='viridis', alpha=0.9, edgecolor='k', linewidth=0.2)
       ax2.set_xlabel("Theta0 Shift", labelpad=12)
       ax2.set_ylabel("Theta1 Multiplier", labelpad=12)
       ax2.set_zlabel("MSE", labelpad=15)
       ax2.set_title("MSE Surface", pad=20)
       ax2.view_init(elev=35, azim=135)
       ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
       ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
       ax2.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
       fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1).set_label('MSE')

       # ---- RMSE subplot ----
       ax3 = fig.add_subplot(1, 3, 3, projection='3d')
       surf3 = ax3.plot_surface(T0, T1, RMSE_surface, cmap='cividis', alpha=0.9, edgecolor='k', linewidth=0.2)
       ax3.set_xlabel("Theta0 Shift", labelpad=12)
       ax3.set_ylabel("Theta1 Multiplier", labelpad=12)
       ax3.set_zlabel("RMSE", labelpad=15)
       ax3.set_title("RMSE Surface", pad=20)
       ax3.view_init(elev=35, azim=135)
       ax3.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
       ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
       ax3.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
       fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, pad=0.1).set_label('RMSE')

       plt.tight_layout(w_pad=3)
       plt.show()
#takes colls,targetvar returns regressionmodel
def regression(y,data):
       #for a regression model I need to know the number of datapoints I'm using (multiple vs linear)
       #then I need to know the target variable & the data in a separate structure & form them into a train/test split
       #then use linear regression of find the intercept & coefficient
       
       
       #data = pd.DataFrame(data[1])
       print("numcols = ", len(data.columns))
       print("len of y = ", len(y))
       print("len of x = ", len(data))
       x_train, x_test, y_train, y_test = train_test_split(data, y,test_size=0.25, random_state = 42)
       reg = LinearRegression()
       reg.fit(x_train,y_train)
       if len(data.columns) == 1:

              print(reg.intercept_)
              print(reg.coef_)
              predictions = reg.predict(x_test)


       else:
              #multiple linear regression

              print(reg.intercept_)
              print(reg.coef_)
              predictions = reg.predict(x_test)
              #np.argmax(model.predict(x_val), axis=-1)
       #check to see if linear regression has a good enough result, if it does not then preform polynomial regression
       print(predictions)
       print(predictions.dtype)
       if y_test.dtype == "int64":
              predictions = predictions.astype(np.int64)
       print(y_test.dtype)
       accuracy = np.sqrt(metrics.mean_squared_error(y_test, predictions))
       print("accuracy lin = ", accuracy_score(y_test, predictions))
       x_train, x_test, y_train, y_test = train_test_split(data, y,test_size=0.25, random_state = 42)
       polyreg = LinearRegression()
       #polynomial regression
       degree = len(x_train.columns)

       print(x_train)
       print("degree = ", degree)
       poly = PolynomialFeatures(degree)
       x_train_poly = poly.fit_transform(x_train)
       
       polyreg.fit(x_train_poly,y_train)
       x_test_poly = poly.fit_transform(x_test)
       predictions = polyreg.predict(x_test_poly)
       if y_test.dtype == "int64":
              predictions = predictions.astype(np.int64)
       polyaccuracy = np.sqrt(metrics.mean_squared_error(y_test, predictions))

       print("accuracy pol = ", accuracy_score(y_test, predictions))
       if polyaccuracy < accuracy:
              return polyreg, polyaccuracy
       return reg, accuracy
#loop regression until I have a decent score, also don't use collision_severity. Mainly check for the best mincorr

def knnmodel(y,data,k):

       #appends models in the format [model, y]
       ovamodelarr = [] #dict of the class & related one v all model
       accuracyarr = []
       #this works by selecting only a specific class 
       x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.25, random_state=42)
       for cls in np.unique(y_train):
              y_bin = (y_train == cls).astype(int) 
              y_bin_test = (y_test == cls).astype(int)
              #this creates essentially a boolean 0 or 1 out of the classes by replacing all positions where y_train = cls with 1 & all positions where this is false with 0
              clf = knn(n_neighbors=k)
              #then we train a model based on this boolean classifier
              clf.fit(x_train, y_bin)
              accuracyarr.append(clf.score(x_test,y_bin_test))
              ovamodelarr.append((cls, clf))
       #I now have a list of models that can detect if an input is in class y or not in class y
       #this is then applicable by iterating through the model and finding which has the highest predicted certainty for the input being in class y

       #to loop this array to find the best k value I need to return an individual number for the accuracy score where higher = better
       #I can do this by taking the mean model accuracy of my array and returning this. Then I have a reasonably accurate idea of the accuracy of my models.
       #its likely better to return the full array of accuracies & models and deal with it in the menu function as I may have uses for the accuracy array over the mean accuracy
       print(sum(accuracyarr))
       print(len(accuracyarr))
       modelscore = sum(accuracyarr)/len(accuracyarr)
       print("modelscore = ", modelscore)
       #model = knn(n_neighbors = k)
       #model.fit(x_train,y_train)
       #modelscore = model.score(x_test,y_test)

       #indata = feature1, feature2... + preprocessing values (pcanum)
       #apply preprocessing to indata
       #loop through the model array & get the 1 or 0 class for each one
       #then return the correct model
       
       return ovamodelarr,accuracyarr
#one vs all and one vs one are training methods
#therefore I can use these methods within my output for SVM & knn models
def linmodel(y,data,c,gamma):
       pass
def polymodel(y,data,c,gamma,degree):
       pass
def polymodel(y,data,c,gamma):
       pass
def svmmodel(y,data):
       print(y)
       #this is the binary svm model which classifies between two classes
       print(y)
       x_test, x_train,y_test,y_train = train_test_split(y,data,test_size = 0.25, random_state = 42)
       x_train = pd.DataFrame(x_train)


# I'm confused about where in my code to add the grid search. Since I'm confused how it fits in alongside ovo & ova
       param_grid = {'C': [0.1,1,10,100],
                     'gamma': [1,0.1,0.01,0.001],
                     'kernel': ['linear','poly','rbf'],
                     'degree': [2,3,4]}
       svm = SVC( class_weight = 'balanced', probability= True )
       grid_search = GridSearchCV(
       estimator=svm,
       param_grid=param_grid,
       cv=5,
       scoring="accuracy",
       verbose=1,
       return_train_score=True)
       grid_search.fit(x_train, y_train)
       best_model = grid_search.best_estimator_
       best_params = grid_search.best_params_
       best_kernel = best_params['kernel']
       best_C = best_params['C']
       best_gam = best_params['gamma']
       best_deg = best_params['degree']

       if best_kernel == 'linear':
              #classes = np.unique(y_train)
              #ova_classifiers = []
              #for cls in classes:
              #       y_bin = (y_train == cls).astype(int)
              #       clf = SVC( kernel = kernel, class_weight = 'balanced', gamma = gamma )
              #       clf.fit(x_train, y_bin)
              #       ova_classifiers.append((cls, clf))

              #classes = np.unique(y_train)
              #ovo_classifiers = []
              #for cls1, cls2 in combinations(classes, 2):
              #       idx = np.where((y_train == cls1) | (y_train == cls2))[0]
              #       X_pair = x_train[idx]
              #       y_pair = y_train[idx]
              #       y_bin = (y_pair == cls1).astype(int)
              #       clf = SVC( kernel = kernel, class_weight = 'balanced', gamma = gamma)
              #       clf.fit(X_pair, y_bin)
              #       ovo_classifiers.append(((cls1, cls2), clf))

              svm = SVC(C=best_C, kernel = best_kernel, class_weight = 'balanced', gamma = best_gam )
       elif best_kernel == 'polynomial':
              #these do all use ovo & ova in the same way, there are only differences in the arguments that are needed to be passed.
              #could change ovo & ova into their own function like the example code
              svm = SVC(C=best_C, kernel = best_kernel, class_weight = 'balanced', gamma = best_gam, degree = best_deg)
       elif best_kernel == 'rbf':
              svm = SVC(C=best_C, kernel = best_kernel, class_weight = 'balanced', gamma = best_gam )
       svm.fit(x_train,y_train)
       
       y_pred = best_model.predict(x_test)
       accuracy = accuracy_score(y_pred,y_test)
       return best_model,accuracy
def getinput(features, colls):
       entryarr = []
       x = 0
       while x < len(features):
              entry_dtype = colls[features[x]].dtype
              try:
                     if entry_dtype == 'float64':
                            entryarr.append(float(input(print("enter the value for ", features[x], ":\n"))))
                            
                     elif entry_dtype == 'int64':
                            entryarr.append(int(input(print("enter the value for ", features[x], ":\n"))))
                     elif entry_dtype == 'str':
                            entryarr.append(int(input(print("enter the value for ", features[x], ":\n"))))
                     else:
                            print("unknown data type = ", entry_dtype)
              except:
                     print("invalid datatype, please use ", entry_dtype)
                     continue
              
              x +=1
       return entryarr

#ovo = one vs one
#ova = one vs all
#input:
# modeltype,model, data preprocessing info (pcacount, selected features), ovo / ova, user input, model accuracy
def applymodel(inputtype,inputs,model):
       #impliment 1v1 & 1vall
       #user enters a list of data. For this I need all fields & how I processed the test data
       #then I can process the user input the same as the test data
       #then I can predict using one vs one or one vs all based on parameters
       #I can do this prediction when testing the accuracy of the model and return if one vs one or one vs many is more accurate

       if model in ["svm", "knn"]:


       
              pass
def menu(colls):
       choice = 1#int(input("enter the option you want:\n 1) create supervised model"))
       mincorr= 0.06
       slow = False
       targetvar = 0
       choice = 0
       mincorr = 0.1
       k = 5
       while targetvar not in (colls.columns.values) and choice not in [1,2,3,4]:
              #print(stringcolls.columns)

              #targetvar = input("Enter the name of the collumn you want to predict, options:\n" + str(colls.columns) + "\n")
              #choice = int(input("if you are trying to predict a class based on input data enter 1 for neighbors model or 2 for SVM, enter 3 for a regression model\n"))
              targetvar = "collision_severity"
              choice = 2
       #if the user wants to enter a string y point then use a correlation model
      

              if choice == 1:
                     while True:
                            mincorrdict = {}
                            corrpointer = mincorr
                            #add features return from preprocessing, post mincorr find
                            y,data,features,pca,scaler  = preprocessing(colls,targetvar,mincorr)
                            kmodel,accuracy= knnmodel(y,data,k)
                            mincorrdict[mincorr] = accuracy
                            while True:
                                   
                                   kdict = {}
                                   kpointer = k
                                   kmodel,accuracy= knnmodel(y,data,k)
                                   kdict[k] = accuracy
                                   for i in np.arange(1,20,1):
                                          kmodel,accuracy= knnmodel(y,data,k+i)

                                          kdict[k+i] = accuracy
                                          if k-i < 1:
                                                 print("continue")
                                                 continue
                                          kmodel,accuracy= knnmodel(y,data,k-i)

                                          kdict[k-i] = accuracy

                                   k = max(kdict, key=kdict.get)
                                   print("K = ", k)
                                   if k == kpointer:
                                          print("KBREAK")
                                          break
                            for i in np.arange(0.00,0.1,0.01):
                                   try:
                                          y,data,features,pca,scaler  = preprocessing(colls,targetvar,mincorr)
                                   except ZeroDivisionError:
                                          print("this iteration has zero columns within the correlation range\n mincorr = ", mincorr+i)
                                          continue
                                   kmodel,accuracy= knnmodel(y,data,k)
                                   mincorrdict[mincorr+i] = accuracy 
                                   try:
                                          y,data,features,pca,scaler = preprocessing(colls,targetvar,mincorr)
                                   except ZeroDivisionError:
                                          print("this iteration has zero columns within the correlation range\n mincorr = ", mincorr-i)
                                          continue
                                   kmodel,accuracy= knnmodel(y,data,k)
                                   mincorrdict[mincorr-i] = accuracy

                                   #converting so this value is better the higher the number to prevent needing a separate for loop depending on the model type
                                   #move findmincorr into menu, then I always know which model to loop through without having to individually hardcode each model into 
                                   #get an array of models which I can pass into mincorr to iterate over them
                                   #need to get expected input number for the regression model to predict
                                   #I could also add a check for best degree after the mincorrs code
                                   #BUT I'M ONTO SVM NOW

                            mincorr = max(mincorrdict, key=mincorrdict.get)
                            print("mincorr = ", mincorr)
                            if mincorr == corrpointer:
                                   break
                     y,data,features,pca,scaler = preprocessing(colls,targetvar,mincorr)
                     kmodel,accuracy= knnmodel(y,data,k)
                     x = 0
                     #entryarr = []
                     #while x < len(features):
                     #       entry_dtype = colls[features[x]].dtype
                     #       try:
                     #              if entry_dtype == 'float64':
                     #                     entryarr.append(float(input(print("enter the value for ", features[x], ":\n"))))
                     #                     
                     #              elif entry_dtype == 'int64':
                     #                     entryarr.append(int(input(print("enter the value for ", features[x], ":\n"))))
                     #              elif entry_dtype == 'str':
                     #                     entryarr.append(int(input(print("enter the value for ", features[x], ":\n"))))
                     #              else:
                     #                     print("unknown data type = ", entry_dtype)
                     #       except:
                     #              print("invalid datatype, please use ", entry_dtype)
                     #              continue
                     #       
                     #       x +=1
                     entryarr = getinput(features,colls)
                     indf = pd.DataFrame([entryarr], columns = features)
                     resultarr = []
                     result = -1
                     

                     # Scale the input using the scaler used during training
                     indf = scaler.transform(indf)
                     indf = pca.transform(indf)


                     # Predict using best_model from GridSearchCV
                     #prediction = best_model.predict(user_input_scaled)[0]
                     #prediction_proba = best_model.predict_proba(user_input_scaled)[0]
                     for i in range(0,len(kmodel)):
                            print("kmodeli = " , kmodel[i][1])
                            resultarr.append([kmodel[i][1].predict(indf)[0], accuracy[i]])
                     for i in range(0,len(resultarr)):
                            if resultarr[i][0] == 1:
                                   print("this class has been found", kmodel[i][1])
                                   if result == -1:
                                          result = [kmodel[i][0], accuracy[i]]
                                   else: #two one vs alls are true, this item has been classified into both classes
                                          if accuracy[i] > result[1]:
                                                  result = [kmodel[i][0], accuracy[i]]
                     print("your final result is:\n", result[0], "\nwith an accuracy of:\n", result[1])
                                   

              elif choice == 2:
                     y,data,features,pca,scaler  = preprocessing(colls,targetvar,mincorr)
                     thismodel, accuracy = svmmodel(y,data)
                     entryarr = getinput(features,colls)
                     indf = pd.DataFrame([entryarr], columns = features)
                     indf = pca.transform(indf)
                     result = thismodel.predict(indf)
                     print("your final result is:\n", result, "\nwith an accuracy of:\n", accuracy)
                     #get a small grid for values
                     #search for mincorr alongside grid



              elif choice == 3:
                     mincorrdict = {}
                     pointer = mincorr
                     for i in np.arange(0.00,0.1,0.01):
                            try:
                                   y,data,features,pca,scaler = preprocessing(colls, targetvar,mincorr+i)
                            except ZeroDivisionError:
                                   print("this iteration has zero columns within the correlation range\n mincorr = ", mincorr+i)
                                   continue
                            regmodel,rmse = regression(y,data)
                            mincorrdict[mincorr+i] = 1-rmse #converting so this value is better the higher the number to prevent needing a separate for loop depending on the model type
                            try:
                                   y,data,features,pca,scaler = preprocessing(colls, targetvar,mincorr-i)
                            except ZeroDivisionError:
                                   print("this iteration has zero columns within the correlation range\n mincorr = ", mincorr-i)
                                   continue
                            regmodel,rmse = regression(y,data)
                            mincorrdict[mincorr-i] = 1-rmse 
                            #converting so this value is better the higher the number to prevent needing a separate for loop depending on the model type
                            #move findmincorr into menu, then I always know which model to loop through without having to individually hardcode each model into 
                            #get an array of models which I can pass into mincorr to iterate over them
                            #need to get expected input number for the regression model to predict
                            #I could also add a check for best degree after the mincorrs code
                            #BUT I'M ONTO SVM NOW

                     mincorr = max(mincorrdict, key=mincorrdict.get)
                     if mincorr == pointer:
                            break
                     entryarr = getinput(features,colls)
                     indf = pd.DataFrame([entryarr], columns = features)
                     resultarr = []
                     result = -1
                     

                     # Scale the input using the scaler used during training
                     indf = scaler.transform(indf)
                     indf = pca.transform(indf)


                     # Predict using best_model from GridSearchCV
                     #prediction = best_model.predict(user_input_scaled)[0]
                     #prediction_proba = best_model.predict_proba(user_input_scaled)[0]
                     result = []
                     result.append(regmodel.predict(indf))
                     result.append(rmse)
                     print("your final result is:\n", result[0], "\n +- :\n", result[1])
              elif choice == 4:
                     model,rmse, targetvar,degree,x_name = getbestdegree(colls,corr_matrix,mincorr,numcolls,targetvar,3,5,5,slow)
                     print("accuracy: " + str(rmse))
                     model.predict(np.array([numcolls[x_name].mean()]))
                     break

menu(colls)
#menu(preprocessed_colls)
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
