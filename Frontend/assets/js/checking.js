var flag=0;
var msg;
var message;
var varPrediction;
const loader = document.getElementById('loader');
const Mail = document.getElementById('Mail');

function ValidateMail()
{
    message=document.getElementById("message").value;
    //console.log(message);
    var lblError1 = document.getElementById("lblError1");
    lblError1.innerHTML = "";
    if(message=="")
    {
        lblError1.style.color='red';
        lblError1.innerHTML='Please enter the mail.';
        flag=1;
    }
    else
    {
        flag=0;

    }
        
        enableButton();

            
    }

function enableButton()
{
    if(flag==0)
        {
             document.getElementById("button").disabled = false;
             console.log(message);
             //sendServer(msg);

        }
        else if (flag==1) 
        {
            
            document.getElementById("button").disabled = true;
        }

}
  var flag1=0;
 var Alert  = new CustomAlert();
var detectRate=0;
function putData()
{
   
   sendServer(message);
}




function CustomAlert(){

  this.render = function()
  {
      //Show Modal
var popUpBox = document.getElementById('popUpBox');

      if (detectRate>=70)
      {
         document.getElementById("result").src="images/report.jpg";
        var report= document.getElementById("wresult");
        report.innerHTML = "Report: Phishing Detect with Detection Rate of : "+detectRate;
       // report.innerHTML = "'Report: Phishing Detect with Accuracy of '";
         varPrediction=0;
      }        
      
      else if(detectRate<70 && detectRate>50)
      {
        document.getElementById("result").src="images/report3.jpg";
        var report= document.getElementById("wresult");
        report.innerHTML = "Report: Suspected Phishing Email With Detection Rate of : "+detectRate;
         varPrediction=0;
    }

    else
    {
       document.getElementById("result").src="images/report2.png";
        var report= document.getElementById("wresult");
        report.innerHTML = "Report: Non Phishing Detect with Accuracy of : "+detectRate;
         varPrediction=0;
    }

      popUpBox.style.display = "block";

      //Close Modal

      document.getElementById('closeModal').innerHTML = '<button id= "Popup-ok" onclick="Alert.ok()">OK</button>';
        
    }
this.ok = function(){
  document.getElementById('popUpBox').style.display = "none";
  document.getElementById('popUpOverlay').style.display = "none";
}
}



function sendServer(message)
{
   var myObj = { 
                 "MailText":message
               }
              console.log(myObj);

$.ajax({
    url: 'http://localhost:5000/checkMail',
    method: 'POST',
    //dataType: 'json',
    data: JSON.stringify(myObj),
    headers: 
    {
        'Content-Type':'application/json',
        'Accept':'text'
    },


   beforeSend: function()
   {
    // Show image container
        $("#loader").show();


    $("#Mail").hide();
   },
  
  
    success : function(response)
  {
    varPrediction=response;
   
    $("#loader").hide();
    detectRate = parseInt(varPrediction);
              //console.log(response)
              console.log(detectRate);
              Alert.render();
     
     $("#Mail").show();

    //location.replace("error.html");
    //alert("Registered Successfully");
    
  },
error : function(jqXHR, exception){
  if(jqXHR.status == 400){
    alert("Issue With Model.. Try Again");
    //location.replace("login.html")
      $("#loader").hide();
    $("#Mail").show();

  }
  else{
    alert("There is an issue on the server. Please contact system administrator.");
      $("#loader").hide();
    $("#Mail").show();
  }

},
  

  });
}