var check = function() 
{
  if (document.getElementById('password').value ==
    document.getElementById('cpassword').value) {
    document.getElementById('message').style.color = 'green';
    document.getElementById('message').innerHTML = 'matching';
  } else {
    document.getElementById('message').style.color = 'red';
    document.getElementById('message').innerHTML = 'not matching';
  }
  
}


let flag=0;
let user;
let ln;
let pwd;
let cpwd;
let mail;
let ph;

var valid= function()
{
    registration()
}


function registration() 
{
 user=document.getElementById('Username').value;
 ln=document.getElementById('lname').value;
 pwd=document.getElementById('password').value;
 cpwd=document.getElementById('cpassword').value;
 mail=document.getElementById('email').value;
 ph=document.getElementById('phone').value;

flag=0;


if (flag==0) 
{
if(user.length == 0)
{
flag=1;
}

if(ln.length == 0)
{
flag=1;
}

 if(mail.length == 0)
{
flag=1;
}

if(ph.length == 0)
{  
flag=1;
}

}

sendObj()

}

function sendObj()
{

if(flag==0)
{
	var myObj = {	
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
		//location.replace("login.html");
		 //window.location="file:///C:/Users/HP/Desktop/website/Frontend/login.html"

        alert("Registered Successfully");
		window.location="file:///C:/Users/HP/Desktop/website/Frontend/login.html";
	},
error : function(jqXHR, exception){
	if(jqXHR.status == 400){
		alert("Cannot Register You, Please Enter New email and password");
		//location.replace("registration.html")
	   window.location="file:///C:/Users/HP/Desktop/website/Frontend/registration.html"
    }
	else{
		alert("There is an issue on the server. Please contact system administrator.");
	}

},
	

  });



}


}