import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

from Config import Config
from training.ModelSelection import ModelSelection
from training.PreProcessing import PreProcessing
from FileOperation import FileOperation


class Training:
    def __init__(self):
        # self.traindata = datapath
        self.fileops = FileOperation()
        self.config = Config()
        print("training has begin...")

    # def preprocessing(self):
    #     preprocess = PreProcessing(self.traindata)
    #     self.traindata = preprocess.drop_LOANID()
    #     self.traindata = preprocess.cleanData()
    #     return preprocess.split_data(self.traindata)

    def trainModel(self):

        loaded_data = self.fileops.loadModel("preprocessed_data", self.config.preprocesseddatapath)
        if (loaded_data is None):
            print("error loading data , please check/retry")
            return None
        else:
            xtr, xte, ytr, yte = loaded_data[0], loaded_data[1], loaded_data[2], loaded_data[3]
            modelselection = ModelSelection()
            bestmodel = modelselection.bestModel(xtr, ytr, xte, yte)
            # bestmodel = modelselection.decisionTree(xtr,ytr)
            savepath = self.config.modelSavePath
            modelSaved = self.fileops.saveModel(bestmodel, "model_1", savepath)
            if (modelSaved):
                print("Model saved successfully")
            else:
                print("Model cannot be saved")

            return modelSaved
            # model = self.fileops.loadModel("model_1", savepath)
            #
            # ypred = model.predict(xte)
            # acc = accuracy_score(yte, ypred)
            # print(acc)
            # return str(acc)
