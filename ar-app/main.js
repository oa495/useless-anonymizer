var videoElm = document.querySelector('video');
// Make sure the video fits the window.
var constrains = { video: { mandatory: { minWidth: window.innerWidth }}};

if (navigator.getUserMedia) {
    navigator.getUserMedia(constrains, function(stream) {
        videoElm.src = window.URL.createObjectURL(stream);
        // When the webcam stream is ready get it's dimensions.
        videoElm.oncanplay = function() {
            init(videoElm.clientWidth, videoElm.clientHeight);
            // Init everything ...

            requestAnimationFrame(render);
        }
    }, function() {});
}

var history = [];
var ws = new WebSocket('ws://localhost:9000');

ws.onopen = function() {
    console.log('onopen');
};

ws.onmessage = function (event) {
    var m = JSON.parse(event.data);
    console.log(m);
};