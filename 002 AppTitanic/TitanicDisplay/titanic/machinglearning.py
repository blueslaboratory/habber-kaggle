import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class MachingLearning:
    def __init__(self):
        def clean(data):
            data["Sex"] = data["Sex"].replace({'male': 1, 'female': 0})
            data["Age"].fillna(data["Age"].median(), inplace=True)
            data["Relatives"] = data["SibSp"] + data["Parch"]

            data = data.drop(["Name", "Ticket", "Cabin", "Embarked", "Fare", "SibSp", "Parch"], axis=1)
            return data

        # pd.set_option("display.max_columns", None)

        train_data = pd.read_csv("./titanic/data/train.csv")
        train_data = clean(train_data)

        print(train_data.head(1))

        test_data = pd.read_csv("./titanic/data/test.csv")
        test_data = clean(test_data)

        survived = train_data["Survived"]

        fields = ["PassengerId", "Pclass", "Sex", "Age", "Relatives"]

        data_train = pd.get_dummies(train_data[fields])
        data_test = pd.get_dummies(test_data[fields])

        # # Utilizando random tree
        # modelRandomTree = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
        # modelRandomTree.fit(data_train, survived)
        #
        # predictions = modelRandomTree.predict(data_test)
        #
        # output = pd.DataFrame({'PassengerId': test_data.PassengerId,
        #                        'Survived': predictions,
        #                        'Pclass': test_data.Pclass,
        #                        'Sex': test_data.Sex,
        #                        'Age': test_data.Age,
        #                        'Relatives': test_data.Relatives})
        # output.to_csv('./titanic/data/submission_RandomTree.csv', index=False)
        # print("Your submission was successfully saved!")

        # Utilizando Support Vector Machine
        model_svm = SVC(kernel='linear', C=1, gamma='auto', random_state=1)
        model_svm.fit(data_train, survived)

        predictions = model_svm.predict(data_test)

        output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                               'Survived': predictions,
                               'Pclass': test_data.Pclass,
                               'Sex': test_data.Sex,
                               'Age': test_data.Age,
                               'Relatives': test_data.Relatives})
        output.to_csv('./titanic/data/submission_SVM.csv', index=False)
        print("Your submission was successfully saved!")
