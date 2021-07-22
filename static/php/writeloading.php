<?php
    // log in vao database
    include("config.php");
    // doc user input
    $red = $_POST["r"];
    $green = $_POST["g"];
    $blue = $_POST["b"];
    // update lai database
    $sql = "update rgbValues set red=$red,green=$green,blue=$blue";
    mysqli_query($conn, $sql);
    mysqli_close($conn);
?>