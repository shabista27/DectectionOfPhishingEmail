
import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="mail-server"
)


def logins(user,pwd):
        print(user)
        mycursor = mydb.cursor()

        mycursor.execute('SELECT * FROM userlogin WHERE Username = %s AND password = %s', (user, pwd))
        myresult = mycursor.fetchall()
        print(myresult)
        
        if myresult:
            msg=True
        else:
            msg=False
        return msg
    


def registrations(fname,lname,pwd,email,ph):
    print("Going to database")
    mycursor = mydb.cursor()
    try:
        sql = "INSERT INTO `registration` (`Id`, `First Name`, `Last Name`, `Password`, `Email`, `Phone Number`) VALUES ('NULL', %s, %s, %s, %s, %s)"
        val = (fname,lname,pwd,email,ph)
        mycursor.execute(sql,val)
        mydb.commit()
     
        print("Inserted")
        return True
    
    except mysql.connector.Error as err:
        #print(mycursor.execute(sql, val))
        print("Message", err.msg)
        return False
        
def rating(rating):
    print("Going to Database")
    mycursor=mydb.cursor()
    try:
        sql = "INSERT INTO `feedback` (`Id`, `Rating`) VALUES ('NULL', '"+rating+"')"
        mycursor.execute(sql)
        mydb.commit()
        
        print("Inserted")
        return True
    except mysql.connector.Error as err:
        print("Message",err.msg)
        return False        