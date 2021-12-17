from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold,cross_val_score,GridSearchCV,cross_val_predict
from sklearn.tree import DecisionTreeClassifier


class ModelSelection:

    def __init__(self):
        self.scoring = "precision"

    def randomForest(self,xtr,ytr):

        rfc = RandomForestClassifier()
        param = {'n_estimators': [50, 100, 200], 'criterion': ['entropy', 'gini'],
                 'max_features': ["auto", 'sqrt', 'log2', 'None']}

        rfg = GridSearchCV(rfc,param_grid=param,scoring=self.scoring)
        print("gridsearch for randomforest")
        rfg.fit(xtr, ytr)

        return rfg.best_estimator_

    def gradientBoostd(self,xtr,ytr):
        gbc = GradientBoostingClassifier()
        param = {'learning_rate': [0.01, 0.1, 0.5], 'criterion': ['friedman_mse', 'mse', 'mae'],
                 'max_features': ["auto", 'sqrt', 'log2', 'None']}

        gbg = GridSearchCV(gbc,param_grid=param,scoring=self.scoring)
        print("gridsearch for gradient boost")
        gbg.fit(xtr, ytr)
        return gbg.best_estimator_

    def bestModel(self,xtr,ytr,xte,yte):

        randomforest = self.randomForest(xtr,ytr)
        gradientboost = self.gradientBoostd(xtr,ytr)

        randomforest_ypred = randomforest.predict(xte)
        randomforest_acc = accuracy_score(randomforest_ypred,yte)

        gradientboost_ypred = gradientboost.predict(xte)
        gradientboost_acc = accuracy_score(gradientboost_ypred,yte)

        if(gradientboost_acc > randomforest_acc):
            print("gradient boost gives better performance")
            best_model = gradientboost
        else:
            print("random forest gives better performance")
            best_model =randomforest

        return best_model

    def decisionTree(self,xtr,ytr):
        dtc = DecisionTreeClassifier()
        dtc.fit(xtr,ytr)
        return dtc

