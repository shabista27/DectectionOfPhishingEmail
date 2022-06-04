var flag=0;

var name;
var mail;
var rate;
var msg;

function ValidateMessage()
{
    var message=document.getElementById("message").value;
    var lblError1 = document.getElementById("lblError1");
    lblError1.innerHTML = "";
    if(message=="")
    {
        lblError1.style.color='red';
        lblError1.innerHTML='If there is no msg then write NA. ';
        flag=1;
         enableButton(); 
    }
    else
    {
        flag=0;
        msg=message;
        
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
        enableButton();

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

function feedback()
{
             name=document.getElementById("name").value;
             
             rate=document.querySelector('input[name="rating"]:checked').value
             
           

 var myObj = 
    {   
        "Name":name,
        "Rating":rate,
        "Email":mail,
        "Message":msg

    }
console.log(myObj);


$.ajax({
      url: 'http://localhost:5000/feedback',
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
        location.replace("index.html");
        alert("Feedback Recorded");
        
    },
error : function(jqXHR, exception){
    if(jqXHR.status == 400){
        alert("Something went wrong.. Check the details");
        
    }
    else
    {
        alert("There is an issue on the server. Please contact system administrator.");
    }


},
    

  });





}


