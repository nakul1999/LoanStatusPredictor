import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

from Config import Config
from training.ModelSelection import ModelSelection
from training.PreProcessing import PreProcessing
from FileOperation import FileOperation


class Training:
    def __init__(self, datapath):
        self.traindata = datapath
        print("training has begin...")

    def preprocessing(self):
        preprocess = PreProcessing(self.traindata)
        self.traindata = preprocess.drop_LOANID()
        self.traindata = preprocess.preprocessing()
        return preprocess.split_data(self.traindata)

    def model_training(self, xtr, xte, ytr, yte):
        modelselection = ModelSelection()

        bestmodel = modelselection.bestModel(xtr, ytr,xte,yte)
        fileOps = FileOperation()
        configs = Config()
        savepath = configs.modelSavePath
        fileOps.saveModel(bestmodel, "model_1", savepath)
        model = fileOps.loadModel("model_1", savepath)

        ypred = model.predict(xte)
        acc = accuracy_score(yte, ypred)
        print(acc)
        return str(acc)
