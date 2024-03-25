from flask import Flask, render_template, url_for, flash, redirect
import joblib
import numpy as np
import pandas as pd
from flask import request

import datetime
import matplotlib.pyplot as plt
import yfinance
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#ML model
from keras.models import load_model

app = Flask(__name__, template_folder='templates')
global result,prediction,Color
result=-1

Scaler=StandardScaler()
date=datetime.date.today()

@app.route("/")
def index():
    return render_template("index.html",date=date)

@app.route("/dashboard")
def dashboard():
    return render_template("index.html",date=date)


@app.route("/health_info")
def health_info():
    return render_template("health/health_info.html")

@app.route("/Heart")
def Heart():
    return render_template("health/heart.html")

def Heart_Value_Predictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==7):
        loaded_model = joblib.load(r'D:\PROJECTS\TNSDC-Python Hackathon Project\FinFit AI\Models\health\heartDisease_model.pkl')
        std=Scaler.fit_transform(to_predict)
        result = loaded_model.predict(std)
    return result[0]

@app.route('/heart_predict', methods = ["POST"])
def heart_predict():
    global result,prediction,Color
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        #diabetes
        if(len(to_predict_list)==7):
            result =Heart_Value_Predictor(to_predict_list,7)
    
    if(int(result)==1):
        prediction = "Sorry you chances of getting the disease. Please consult the doctor immediately"
        Color = "danger"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
        Color = "success"
    return(render_template("health/heart.html", prediction_text=prediction,color=Color))       

@app.route("/cancer")
def cancer():
    return render_template("health/cancer.html")

def Cancer_Value_Predictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==5):
        loaded_model = joblib.load(r'D:\PROJECTS\TNSDC-Python Hackathon Project\FinFit AI\Models\health\Brease_Cancer_model.pkl')
        std=Scaler.fit_transform(to_predict)
        result = loaded_model.predict(std)
    return result[0]

@app.route('/Cancer_predict', methods = ["POST"])
def Cancer_predict():
    global result,prediction,Color
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        #cancer
        if(len(to_predict_list)==5):
            result = Cancer_Value_Predictor(to_predict_list,5)
    
    if(int(result)==1):
        prediction = "Sorry you chances of getting the disease. Please consult the doctor immediately"
        Color = "danger"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
        Color = "success"
    return(render_template("health/cancer.html", prediction_text=prediction,color=Color))  

@app.route("/kidney")
def kidney():
    return render_template("health/kidney.html")

def kidney_Value_Predictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==10):
        loaded_model = joblib.load(r'D:\PROJECTS\TNSDC-Python Hackathon Project\FinFit AI\Models\health\kidney_model.pkl')
        std=Scaler.fit_transform(to_predict)
        result = loaded_model.predict(std)
    return result[0]

@app.route('/kidney_predict', methods = ["POST"])
def kidney_predict():
    global result,prediction,Color
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        
        if(len(to_predict_list)==10):
            result = kidney_Value_Predictor(to_predict_list,10)
    
    if(int(result)==1):
        prediction = "Sorry you chances of getting the disease. Please consult the doctor immediately"
        Color = "danger"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
        Color = "success"
    return(render_template("health/kidney.html", prediction_text=prediction,color=Color))       


@app.route("/liver")
def liver():
    return render_template("health/liver.html")

def Liver_Value_Predictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==9):
        loaded_model = joblib.load(r'D:\PROJECTS\TNSDC-Python Hackathon Project\FinFit AI\Models\health\liverDisease_model.pkl')
        std=Scaler.fit_transform(to_predict)
        result = loaded_model.predict(std)
    return result[0]

@app.route('/liver_predict', methods = ["POST"])
def liver_predict():
    global result,prediction,Color
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        
        if(len(to_predict_list)==9):
            result = Liver_Value_Predictor(to_predict_list,9)
    
    if(int(result)==1):
        prediction = "Sorry you chances of getting the disease. Please consult the doctor immediately"
        Color = "danger"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
        Color = "success"
    return(render_template("health/liver.html", prediction_text=prediction,color=Color))


@app.route("/Diabetes")
def Diabetes():
    return render_template("health/diabetes.html")

def Diabetes_Value_Predictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==7):
        loaded_model = joblib.load(r'D:\PROJECTS\TNSDC-Python Hackathon Project\FinFit AI\Models\health\Diabetes_model.pkl')
        std=Scaler.fit_transform(to_predict)
        result = loaded_model.predict(std)
    return result[0]

@app.route('/Diabetes_predict', methods = ["POST"])
def Diabetes_predict():
    global result,prediction,Color
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        
        if(len(to_predict_list)==7):
            result = Diabetes_Value_Predictor(to_predict_list,7)
    
    if(int(result)==1):
        prediction = "Sorry you chances of getting the disease. Please consult the doctor immediately"
        Color = "danger"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
        Color = "success"
    return(render_template("health/diabetes.html", prediction_text=prediction,color=Color))       

@app.route("/finance_info")
def finance_info():
    return render_template("finance/finance_info.html")

@app.route("/health_insurance")
def health_insurance():
    return render_template("finance/health_insurance.html") 

def health_Insurance_Value_Predictor(to_predict_list, size):
    if(size==6):
        loaded_model = joblib.load(r'D:\PROJECTS\TNSDC-Python Hackathon Project\FinFit AI\Models\finance\Insurance_model.pkl')
        result = loaded_model.predict(to_predict_list)
    return result[0]

@app.route('/health_Insurance_predict', methods = ["POST"])
def health_Insurance_predict():
    global result,prediction,Color
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        #new data
    
        data = {'age' : to_predict_list[0],
                'sex' : to_predict_list[1],
                'bmi' : to_predict_list[2],
                'children' : to_predict_list[3],
                'smoker' : to_predict_list[4],
                'region' : to_predict_list[5]
                }

        df = pd.DataFrame(data,index=[0])
        
        if(len(to_predict_list)==6):
            result = health_Insurance_Value_Predictor(df,6)
            result = float("%.2f" %result) 
        
    return(render_template("finance/health_insurance.html", prediction_text=f"Medical Insurance cost for your based on details: ₹. {result:,} /-" ,color="success"))       
    

@app.route("/Loan")
def Loan():
    return render_template("finance/loan.html")

def Loan_Value_Predictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==11):
        loaded_model = joblib.load(r'D:\PROJECTS\TNSDC-Python Hackathon Project\FinFit AI\Models\finance\Loan_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/Loan_predict', methods = ["POST"])
def Loan_predict():
    global result,prediction,Color
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        
        if(len(to_predict_list)==11):
            result = Loan_Value_Predictor(to_predict_list,11)
    
    if(int(result)==1):
        prediction = "Happy to say this,Loan Approved"
        Color = "success"
    else:
        prediction = "Sorry to say this, loan is rejected"
        Color = "danger"
    return(render_template("finance/loan.html", prediction_text=prediction,color=Color)) 


@app.route("/Credit_Card_Score")
def Credit_Card_Score():
    return render_template("finance/credit.html")

def Credit_Card_Score_Predictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==16):
        loaded_model = joblib.load(r'D:\PROJECTS\TNSDC-Python Hackathon Project\FinFit AI\Models\finance\Credit_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/Credit_predict', methods = ["POST"])
def Credit_predict():
    global result,prediction,Color
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        
        if(len(to_predict_list)==16):
            result = Credit_Card_Score_Predictor(to_predict_list,16)
    
    if(int(result)==1):
        prediction = "Your credit card score is Poor"
        Color = "danger"
    elif(int(result)==2):
        prediction = "Your credit card score is Standard"
        Color = "warning"
    else:
        prediction = "Your credit card score is Good"
        Color = "success"
    return(render_template("finance/credit.html", prediction_text=prediction,color=Color)) 

@app.route("/Stock_Market")
def Stock_Market():
    return render_template("finance/Stock_Market.html") 

@app.route("/stock_result",methods=["POST"])
def stock_result():
    if request.method =="POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        stock = to_predict_list[0]

        start="2015-01-01"
        df = yfinance.download(stock,start,date)
        df=df.reset_index()

        #describe data
        dataframe = df.describe()
        dataframe=dataframe.reset_index()
        # get index values
        l =dataframe.columns.values
        index = []
        for i in l:
            index.append(i)

        l = dataframe.values
        data = l.tolist()

        #visualizations
        plt.Figure(figsize=(15,5))
        plt.plot(df.Close)
        plt.savefig(r"D:\PROJECTS\TNSDC-Python Hackathon Project\FinFit AI\static\images\visualization_1.jpg")

        #visualizations-100
        ma100 = df.Close.rolling(100).mean()
        plt.Figure(figsize=(15,5))
        plt.plot(ma100)
        plt.plot(df.Close)
        plt.savefig(r"D:\PROJECTS\TNSDC-Python Hackathon Project\FinFit AI\static\images\visualization_2.jpg")

        #visualizations-200
        ma200 = df.Close.rolling(200).mean()
        plt.Figure(figsize=(15,5))
        plt.plot(ma100)
        plt.plot(ma200)
        plt.plot(df.Close)
        plt.savefig(r"D:\PROJECTS\TNSDC-Python Hackathon Project\FinFit AI\static\images\visualization_3.jpg")


        #split data into train and test
        data_traning = pd.DataFrame(df["Close"][0:int(len(df)*.70)])
        data_testing = pd.DataFrame(df["Close"][int(len(df)*.70)-100:])

        scaler = MinMaxScaler(feature_range=(0,1))
        data_traning_array = scaler.fit_transform(data_traning)


        model = load_model(r"D:\PROJECTS\TNSDC-Python Hackathon Project\FinFit AI\Models\finance\stock_market.h5")

        input_Data = scaler.fit_transform(data_testing)


        x_test = []
        y_test = []

        for i in range(100,input_Data.shape[0]):
            x_test.append(input_Data[i-100:i])
            y_test.append(input_Data[i,0])
            
        x_test,y_test = np.array(x_test),np.array(y_test)

        #making predictions
        y_predicted = model.predict(x_test)

        scale_factor = 1/0.01008726
        y_predicted = y_predicted*scale_factor
        y_test = y_test *scale_factor

        plt.figure(figsize=(15,5))
        plt.plot(y_test,"b")
        plt.plot(y_predicted,"r")
        plt.ylabel("price")
        plt.xlabel("Time")
        plt.legend(["Original Price","Predicated Price"],loc="upper left")
        plt.savefig(r"D:\PROJECTS\TNSDC-Python Hackathon Project\FinFit AI\static\images\predication_fig.jpg")

        return render_template("finance/Stock_Market.html",prediction_text=stock,header=index,rows=data)

@app.route("/bmi")
def bmi():
    return render_template("additional_features/bmi.html")

@app.route("/bmi_result",methods=["POST"])
def bmi_result():
    if request.method =="POST":
        global result,prediction,Color
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        Weight=to_predict_list[0]
        Height=to_predict_list[1]

        Height = (Height/100)
        newbmivalue = float(Weight / (Height*Height))
        if(newbmivalue < 18.6):
            prediction = 'Under weight'
            Color = "warning"
        elif(newbmivalue >= 18.6 and newbmivalue<24.9):
            prediction = 'Normal weight'
            Color = "success"
        else:
            prediction = 'Over weight'
            Color = "danger"
        
        prediction = f"Your body have {prediction} and your body BMI value is {newbmivalue: .2f}"

        return(render_template("additional_features/bmi.html", prediction_text=prediction,color=Color)) 


@app.route("/bmr")
def bmr():
    return render_template("additional_features/bmr.html")

@app.route("/bmr_result",methods=["POST"])
def bmr_result():
    if request.method =="POST":
        global prediction
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        age=to_predict_list[0]
        gender=to_predict_list[1]
        weight=to_predict_list[2]
        height=to_predict_list[3]
        activity=to_predict_list[4]

        if(gender==1):
            if(activity==1):
                newbmrvalue = 1.2 * (66.5 + (13.75 * weight) + (5.003 * height) - (6.755 * age))
            elif(activity==2):
                newbmrvalue = 1.375 * (66.5 + (13.75 * weight) + (5.003 * height) - (6.755 * age))
            elif(activity==3):
                newbmrvalue = 1.55 * (66.5 + (13.75 * weight) + (5.003 * height) - (6.755 * age))
            elif(activity==4):
                newbmrvalue = 1.725 * (66.5 + (13.75 * weight) + (5.003 * height) - (6.755 * age))
            else:
                newbmrvalue = 1.9 * (66.5 + (13.75 * weight) + (5.003 * height) - (6.755 * age))
        else:
            if(activity==1):
                newbmrvalue = 1.2 * (655 + (9.563 * weight) + (1.850 * height) - (4.676 * age))
            elif(activity==2):
                newbmrvalue = 1.375 * (655 + (9.563 * weight) + (1.850 * height) - (4.676 * age))
            elif(activity==3):
                newbmrvalue = 1.55 * (655 + (9.563 * weight) + (1.850 * height) - (4.676 * age))
            elif(activity==4):
                newbmrvalue = 1.725 * (655 + (9.563 * weight) + (1.850 * height) - (4.676 * age))
            else:
                newbmrvalue = 1.9 * (655 + (9.563 * weight) + (1.850 * height) - (4.676 * age))
        
        prediction = f"your body BMR value is {newbmrvalue: .2f}"

        return(render_template("additional_features/bmr.html", prediction_text=prediction,color="danger")) 


@app.route("/diet")
def diet():
    return render_template("additional_features/diet.html")

@app.route('/diet_tracker', methods=['POST'])
def diet_tracker():
        if request.method == "POST":
            to_predict_list = request.form.to_dict()
            to_predict_list = list(to_predict_list.values())
            to_predict_list = list(map(float, to_predict_list))
            breakfast=to_predict_list[0]
            lunch=to_predict_list[1]
            dinner=to_predict_list[2]
            exercise=to_predict_list[3]
            bmr=to_predict_list[4]

            fitness = bmr + exercise - (breakfast + lunch + dinner)

            weekly_deficit = 7 * fitness

            if weekly_deficit > 0 :
                ans = round((weekly_deficit /3600),3)
                result = 'You will lose ' + str(ans) + ' lbs. per week'
            elif weekly_deficit == 0 :
                result = 'Your Weight will stay the same ' 
            else:
                ans = round((-1* weekly_deficit /3600),3)
                result = 'You will gain ' + str(ans) + ' lbs. per week' 
        return render_template('additional_features/diet.html',result=result) 


if __name__ == "__main__" :
    app.run(debug=True,port=9000)


