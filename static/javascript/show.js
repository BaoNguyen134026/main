setInterval(updateImage,1000);
function updateImage(){
    document.getElementById('cim').src = "{{ url_for('video_feed', id='0') }}";
    document.getElementById('gim').src = "{{ url_for('video_feed', id='0') }}";


}