import os
import pickle

class FileOperation:
    def __init__(self):
        pass

    def saveModel(self,model,modelname,savePath):
        modelSaveSucess = False
        try:
            with open(savePath+"/"+modelname+'.sav','wb') as f:
                pickle.dump(model,f)
            modelSaveSucess = True
        except:
            modelSaveSucess = False
        finally:
            return modelSaveSucess

    def loadModel(self,modelName,savePath):

        try:
            with open(savePath+"/"+modelName+".sav","rb") as f:
                data = pickle.load(f)
        except:
            data = None
        finally:
            return data
