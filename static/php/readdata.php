<?php
header('Content-Type: application/json');
$conn = mysqli_connect("localhost","byt","1231","motiontable");

// Doc gia tri RGB tu database
$sql = "select * from motiontable";
$result = mysqli_query($conn,$sql);

$data = array();
foreach ($result as $row){
    $data[] = $row;
}
// add random data
mysqli_query($conn,$sql);
mysqli_close($conn);
echo json_encode($data);
?>