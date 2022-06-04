# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 15:57:42 2021

@author: Shabista
"""
import smtplib
from email.message import EmailMessage


def sendEmail(message, rating, name, email):
        
    msg=EmailMessage()
    msg['Subject']="Name of Sender: "+name+", Rating presented: "+rating 
    msg['From']='Machine Generated'
    msg['To']='intelligence.controlled@gmail.com'
    body="Message: "+message+"\nEmail of Sender: "+email
    msg.set_content(body)
    
    server=smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login("intelligence.controlled@gmail.com","girlpower@3")
    try:
        server.send_message(msg)
        server.quit()
        print("Feedback recorded")
        return True;
    except smtplib.SMTPResponseExceptiont as err:
        print(err.message)
        return False;