import pandas as pd

class MultiCollinearityEliminator():
    def __init__(self, df, target, threshold):
        self.df = df
        self.target = target
        self.threshold = threshold

    def createCorrMatrix(self, include_target = False):
        if include_target:
            corrMatrix = self.df.corr(method = 'pearson', min_periods = 30).abs()
        else:
            temp_df = self.df.drop([self.target], axis = 1)
            corrMatrix = temp_df.corr(method = 'pearson', min_periods = 30).abs()
        return corrMatrix
    
    def createCorrMatrixWithTarget(self):
        corrMatrix = self.createCorrMatrix(include_target = True)                           

        corrWithTarget = pd.DataFrame(corrMatrix.loc[:,self.target]).drop([self.target], axis = 0).sort_values(by = self.target)                    
        print(corrWithTarget, '\n')
        return corrWithTarget

    def createCorrelatedFeaturesList(self):
        corrMatrix = self.createCorrMatrix(include_target = False)                          
        colCorr = []
        for column in corrMatrix.columns:
            for idx, row in corrMatrix.iterrows():                                            
                if(row[column]>self.threshold) and (row[column]<1):
                    if (idx not in colCorr):
                        colCorr.append(idx)
                    if (column not in colCorr):
                        colCorr.append(column)
        print(colCorr, '\n')
        return colCorr
    
    def deleteFeatures(self, colCorr):
        #Obtaining the feature to target correlation matrix dataframe
        corrWithTarget = self.createCorrMatrixWithTarget()                                  
        for idx, row in corrWithTarget.iterrows():
            print(idx, '\n')
            if (idx in colCorr):
                self.df = self.df.drop(idx, axis =1)
                break
        return self.df    
    
    def autoEliminateMulticollinearity(self):
        #Obtaining the list of correlated features
        colCorr = self.createCorrelatedFeaturesList()                                       
        while colCorr != []:
            #Obtaining the dataframe after deleting the feature (from the list of correlated features) 
            #that is least correlated with the taregt
            self.df = self.deleteFeatures(colCorr)
            #Obtaining the list of correlated features
            colCorr = self.createCorrelatedFeaturesList()                                     
        return self.df    