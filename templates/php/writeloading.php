<?php
    // log in vao database
    include("config.php");
    // doc user input
    // update lai database
    $sql = "update loading set red = "0"";
    mysqli_query($conn, $sql);
    mysqli_close($conn);
?>