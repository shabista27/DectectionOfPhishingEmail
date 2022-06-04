# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:35:32 2020

@author: Shabista
"""

import  flask
from flask import Response,request
from flask_cors import CORS
from flask_api import status

from database import registration


app = flask.Flask(__name__)
CORS(app)

@app.route('/registration', methods=['POST'])
def home():
    print(request.json)
    fname = request.json['FirstName']
    lname = request.json['LastName']
    pwd = request.json['password']
    email = request.json['Email']
    ph = request.json['PhoneNumebr']
    print("Going to database")
    resp=registration(fname,lname,pwd,email,ph)
    if resp:
        msg = 'Register successfully !'
        return Response (response=msg, status=status.HTTP_200_OK)
    else:
         msg="Already Register"
         return Response (response=msg, status=status.HTTP_400_BAD_REQUEST)
    


app.run()