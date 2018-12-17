import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.externals import joblib
from sklearn import preprocessing as pre, svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import matplotlib.pyplot as p

# # Needed for plotting using Linux
# p.switch_backend('TKAgg')

# Remove annoying warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

df = pd.read_csv("input/ulsan.csv")


def prediction(type):
    if not type:
        return "Xin kiểm tra loại cần dự đoán"
    else:
        x = np.array(df.drop([type, 'dateTime', 'TIME'], 1))
        y = df[type]

        # split into a training and testing set
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25)

        scaler = pre.StandardScaler().fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        print("Loading model.....")
        # Do not recompute if possible
        try:
            SVR_model = joblib.load('model.pkl')

        except:
            print("Failed, training model.....")
            SVR_model = svm.SVR(kernel='rbf', C=100, gamma=.1).fit(
                x_train_scaled, y_train)
            joblib.dump(SVR_model, 'model.pkl')

        print("Testing the model...")

        predict_y_array = SVR_model.predict(x_test_scaled)
        score = SVR_model.score(x_test_scaled, y_test)

        print("Score of", score)

        print("Plotting the model")

        y_test_np = np.array(y_test[0:100])

        plt.plot(y_test_np, color='g', label="Actual "+type)
        plt.plot(predict_y_array[0:100], color='r', label="Predicted "+type)

        plt.xlabel('Day')
        plt.ylabel(type)
        plt.title(type+' forecast')
        plt.legend()
        plt.show()

        print("Saving the output")
        np.savetxt("predicted_output.csv", np.transpose(
            np.vstack((y_test_np, predict_y_array[0:100]))), fmt="%f", delimiter=",")


ans = True
while ans:
    print("""
    1.Nhiệt độ
    2.Độ ẩm
    3.Tốc độ gió
    4.Hướng gió
    5.Áp suất nước biển
    6.Lượng mưa
    """)
    ans = input("Chọn loại cần dự báo ")
    if ans == "1":
        prediction('TEMPERATURE')
    elif ans == "2":
        prediction('HUMIDITY')
    elif ans == "3":
        prediction('WIND_SPEED')
    elif ans == "4":
        prediction('WIND_DIRECTION')
    elif ans == "5":
        prediction('SEA_LEVEL_PRESSURE')
    elif ans == "6":
        prediction('PRECIPITATION')
    elif ans != "":
        print("\n Not Valid Choice Try again")
