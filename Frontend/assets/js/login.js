function logins() 
{

let user=document.getElementById('Username').value;
let pass=document.getElementById('password').value;
let flag=0;

	if(user.length == 0)
{
alert("Enter the Username");
flag=1;
}

else if(pass.length == 0)
{
alert("Enter the Password");
flag=1;
}

if(flag==0)
{

var myObj = {	
		"Username":user,
		"password":pass
	}
console.log(myObj);



$.ajax({
    url: 'http://localhost:5000/login',
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
		location.replace("https://colab.research.google.com/drive/1WKEhRJSS0d-ikGrcWJFHFypMscUTsJz7#scrollTo=f-x4LeYMKObB");
		//alert("Registered Successfully");
		
	},
error : function(jqXHR, exception){
	if(jqXHR.status == 400){
		alert("Invalid Username and password");
		location.replace("login.html")
	}
	else{
		alert("There is an issue on the server. Please contact system administrator.");
	}

},
	

  });
}
}

