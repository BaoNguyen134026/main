<?php
// Connect to database
$server = "localhost";
$user = "byt"; 
$pass = "1231";
$dbname = "interaction";

$conn = mysqli_connect($server,$user,$pass,$dbname);

// Check connection
if($conn === false){
    die("ERROR: Could not connect. " . mysqli_connect_error());
}
?>