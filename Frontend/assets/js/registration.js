var flag=0;

var user;
var ln;
var pwd;
var mail;
var ph;


function ValidateName()
{
    var name = document.getElementById("name").value;
        var nameError = document.getElementById("nameError");
        nameError.innerHTML = "";
        var expr = /^[A-Za-z]+$/;
        if (!expr.test(name)) 
        {
           nameError.style.color = 'red';
            nameError.innerHTML = 'In Valid Format';
            flag=1;
        }
        else
        {
            nameError.style.color = 'green';
            nameError.innerHTML = 'Valid Format';
            flag=0;
            user=name;
        }

}

function ValidateLname()
{
    var lname = document.getElementById("lname").value;
        var lnameError = document.getElementById("lnameError");
        lnameError.innerHTML = "";
        var expr = /^[A-Za-z]+$/;
        if (!expr.test(lname)) 
        {
           lnameError.style.color = 'red';
            lnameError.innerHTML = 'In Valid Format';
            flag=1;
        }
        else
        {
            lnameError.style.color = 'green';
            lnameError.innerHTML = 'Valid Format';
            flag=0;
            ln=lname;
        }

}


var check = function() 
{
  if (document.getElementById('password').value ==
    document.getElementById('cpassword').value) 
  {
    document.getElementById('message').style.color = 'green';
    document.getElementById('message').innerHTML = 'matching';
    flag=0
    pwd=document.getElementById('password').value;
  } 
  else {
    document.getElementById('message').style.color = 'red';
    document.getElementById('message').innerHTML = 'not matching';
    flag=1;
  }
  
}

 function ValidateEmail()
  {

        var email = document.getElementById("email").value;
        var lblError = document.getElementById("lblError");
        lblError.innerHTML = "";
        var expr = /^([\w-\.]+)@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|(([\w-]+\.)+))([a-zA-Z]{2,4}|[0-9]{1,3})(\]?)$/;
        if (!expr.test(email)) 
        {
            lblError.style.color = 'red';
            lblError.innerHTML = 'Invalid Format';
            flag=1;
            
        }
        else
        {
                lblError.style.color = 'green';
                lblError.innerHTML = 'Valid Format';
                flag=0;
                mail=email;
        }
  }
    

function ValidatePhone()
{
    var phone = document.getElementById("phone").value;
        var lblError1 = document.getElementById("lblError1");
        lblError1.innerHTML = "";
        var expr = /^[6-9]\d{9}$/;
        if (!expr.test(phone)) 
        {
            flag=1;
           lblError1.style.color = 'red';
            lblError1.innerHTML = 'In Valid Format';
            enableButton();
        }
        else
        {
            lblError1.style.color = 'green';
            lblError1.innerHTML = 'Valid Format';
            flag=0;
             ph=phone;
            if(flag==0)
            {
                enableButton();
            }

           
        }
            
        
}

function enableButton()
{
    if(flag==0)
        {
            document.getElementById("button").disabled = false;
            
        }
        else if (flag==1) 
        {
            
            document.getElementById("button").disabled = true;
        }

}

function registration()
{
    var myObj = 
    {   
        "FirstName":user,
        "LastName":ln,
        "password":pwd,
        "Email":mail,
        "PhoneNumber":ph

    }
console.log(myObj);


$.ajax({
      url: 'http://localhost:5000/registration',
    method: 'POST',
    //dataType: 'json',
    data: JSON.stringify(myObj),
    headers: 
    {
        'Content-Type':'application/json',
        'Accept':'text'
    },
    success : function(response)
    {
        //alert(response);
        location.replace("login.html");
        //alert("Registered Successfully");
        
    },
error : function(jqXHR, exception){
    if(jqXHR.status == 400){
        alert("You are already Registered! Please login.");
        location.replace("login.html")
    }
    else{
        alert("There is an issue on the server. Please contact system administrator.");
    }


},
    

  });



}

