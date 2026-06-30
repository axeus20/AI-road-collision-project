def preprocesssing(colls, targetvar,mincorr, predict, pcamin=1):
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
       if not predict:

              #can you use string datatype in a knn classifier?
              colls[targetvar].dropna() #drops all rows where the target variable is null. These rows are not helpful.
              numcolls = colls.select_dtypes(include=['float64', 'int64'])
              stringcolls = colls.select_dtypes(exclude=['float64', 'int64'])
              #split into numcolls & stringcolls as I'm going based on the idea that non numerical features cannot help to classify data using a KNN model.
              #should I use labelling on the string features to transform them into numerical data that can then be used to help classify data?
              numcolls.fillna (numcolls.mean(), inplace=True)
              stringcolls.ffill()
              #the main issue I see with taking a complete mean is that if I am looking for example into collision severity 
              # if I take the mean of all class 1, 2 & 3 and place that into class 1,2 & 3 my dataset may be overwhelmed class 2 data 
              # for example even if the dataset is all class 1 & 3 but the mean falls into class 2 then I will fill all NAN datapoints with data suggesting class 2
              #however this may not be an issue as the model may weight this as 0 if its the mean since there is no variance.
              preprocessed_colls = pd.concat([numcolls, stringcolls], axis=1)
              labeler = LabelEncoder()
              if colls[targetvar].dtype == 'string':
                     y_encoded = labeler.fit_transform(colls[targetvar])
              stringcolls.drop("local_authority_highway_current", axis=1, inplace=True) #without inplace=true
              #print(numcolls.isna().sum())
              #print(stringcolls.isna().sum())
              #split into numbers/non numbers
              corr_matrix = numcolls.corr()
              dfcorrs, corrnames = findcorrx(numcolls,corr_matrix,mincorr,targetvar)
              #remove outliers
              # Source - https://stackoverflow.com/a/23202269
              # Posted by tanemaki, modified by community. See post 'Timeline' for change history
              # Retrieved 2026-06-29, License - CC BY-SA 4.0
              #
              #need to not remove outliers for features that cannot have outliers (classes or years)
              #I have chosen this for my outlier removal method as I could not figure out anything using graphs and I can't see an example using IQR to remove or detect outliers beyond robustscaler
              #I would need information to decide on robust or standardscaler and the only other way to do that is simply directly testing model accuracy
              #  instead of based on an outlier % which I think could be more accurate although it would use the same method of accuracy testing to improve
              #so the next question is, would it be better to simply test standardscaler vs robust scaler for my model as well vs remove those wih too manystandard deviations from the mean? 
              # especially since this would double the time taken, although O notation would be the same
              dfcorrs = dfcorrs[(np.abs(stats.zscore(dfcorrs)) < 3).all(axis=1)] #removes any values at least 3 standard deviations from the mean
   
              #check for data outside IQR to choose robust or minmax scaler. IDK how to check for data quantity outside of iqr
              #choose standardisation 
              MMscaler = mms()
              Sscaler = StandardScaler()
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
                            print(cols1,"---", cols2)
                            print(corr_matrix[cols1][cols2])
                            #the main issue with this pca check is that the function is very arbritrary, however I think it should work to simply scale it to the correct value for the target variable
                            #however it could still cause inacuracy issues & I will likely have to write another function to compare how valuable the reduced dimensionality is compared to marginally higher accuracy.
                            #essentially I'd need to know at what point do I stop improving the accuracy of the model and I can't be sure where that threshold is.
                            pcacheck1 = (abs(corr_matrix[cols1][cols2])-1) * numcolls[cols1].var()
                            pcacheck2 = (abs(corr_matrix[cols1][cols2])-1) * numcolls[cols2].var()
                            print("pcacheck1 = ",pcacheck1)
                            print("pcacheck2 = ", pcacheck2)
                            if (abs(corr_matrix[cols1][cols2])-1) * numcolls[cols1].var() < pcamin or (abs(corr_matrix[cols1][cols2])-1) * numcolls[cols2].var() < pcamin:
                                   pcacount +=1
                                   #this method is trying to combine specific features for PCA, which the second method is letting the PCA function do the work to choose which features combine
                                   #preform PCA
                                   #pca = PCA(n_components=2)
                                   postpcanames.append(cols1)
                                   postpcanames.append(cols2)
                                   #preprocessed_colls.append(pca.fit_transform(numcolls[[cols1, cols2]]))
              print("pcacount = ", pcacount)
              print("postpcanames = ", postpcanames)
              pca = PCA(n_components=pcacount)
              preprocessed_colls = pca.fit_transform(dfcorrs_scaled)

       
       #here is preprocessing based on trying to predict a numberical value based on the data. I will impliment this after understanding the regression & prediction models
       else:
              pass
       return preprocessed_colls
       
