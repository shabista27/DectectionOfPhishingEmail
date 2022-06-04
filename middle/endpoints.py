
"""
Created on Tue Nov 17 20:08:26 2020

@author: Shabista
"""

import flask
from flask import Response,request
from flask_cors import CORS
from flask_api import status

#from flask_api import status

from database import logins
from database import registrations
from database import rating

from loadedModel import detectMail

from feedback import sendEmail


app = flask.Flask(__name__)

CORS(app)



@app.route('/login', methods=['POST'])
def login():
    print(request.json)
    user = request.json['Username']
    pwd = request.json['password']
    resp=logins(user,pwd)
    if resp:
        msg = 'Logged in successfully !'
        return Response (response=msg, status=status.HTTP_200_OK)
    else:
         msg="Invalid Username and Password"
         return Response (response=msg, status=status.HTTP_400_BAD_REQUEST)
    


@app.route('/registration', methods=['POST'])
def registration():
    print(request.json)
    print(type(request.json))
    email = request.json['Email']
    fname = request.json['FirstName']
    lname = request.json['LastName']
    pwd = request.json['password']
    ph = request.json['PhoneNumber']
    print(email,fname,lname,pwd,ph)
    print("Going to database")
    resp=registrations(fname,lname,pwd,email,ph)
    if resp:
        msg = 'Register successfully !'
        return Response (response=msg, status=status.HTTP_200_OK)
    else:
         msg= ' Already Register'
         return Response (response=msg, status=status.HTTP_400_BAD_REQUEST)
    
    

@app.route('/feedback', methods=['POST'])
def feedback():
    print(request.json)
    
    name = request.json['Name']
    rateus = request.json['Rating']
    email=request.json['Email']
    msg=request.json['Message']
    res=sendEmail(msg,rateus,name,email)
    if res:
        print("mail send")
    else:
        print("mail could not be send")
    print(type(rateus));
    #rateus=int(rateus)
    resp=rating(rateus)
    
    if resp:
        msg = 'Feedback Submitted !'
        return Response (response=msg, status=status.HTTP_200_OK)
    else:
         msg="Feedback Cannot be Submitted !"
         return Response (response=msg, status=status.HTTP_400_BAD_REQUEST)
    #message, rating, name, email
    
    
    

@app.route('/checkMail', methods=['POST'])
def checkMail():
   # print(request.json)
    mailText=request.json['MailText']
    
    prediction, resp=detectMail(mailText)
    if resp:
        msg = prediction
        return Response (response=msg, status=status.HTTP_200_OK)
    else:
          msg= prediction
          return Response (response=msg, status=status.HTTP_400_BAD_REQUEST)
    



app.run()
