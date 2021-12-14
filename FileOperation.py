import os
import pickle

class FileOperation:
    def __init__(self):
        pass

    def saveModel(self,model,modelname,savePath):

        with open(savePath+"/"+modelname+'.sav','wb') as f:
            pickle.dump(model,f)

    def loadModel(self,modelName,savePath):

        with open(savePath+"/"+modelName+".sav","rb") as f:
            return pickle.load(f)