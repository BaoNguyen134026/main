function myFunction(imgs) {
    var expandImg = document.getElementById("show_image");
    // var imgText = document.getElementById("imgtext");
    expandImg.src = imgs.src;
    // imgText.innerHTML = imgs.alt;
    expandImg.parentElement.style.display = "block";
  }