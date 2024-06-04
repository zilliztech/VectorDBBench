// The `Streamlit` object exists because our html file includes
// `streamlit-component-lib.js`.
// If you get an error about "Streamlit" not being defined, that
// means you're missing that file.

function sendValue(value) {
  Streamlit.setComponentValue(value)
}



/**
 * The component's render function. This will be called immediately after
 * the component is initially loaded, and then again every time the
 * component gets new data from Python.
 */
function onRender(event) {
  // Only run the render code the first time the component is loaded.
  if (!window.rendered) {
    // You most likely want to get the data passed in like this
    const {height, width, debounce, showControls, startLabel, stopLabel} = event.detail.args

    if (showControls) {
      Streamlit.setFrameHeight(45)
    }

    if (isNaN(height)) {
      height = width / (4/3);
    }

    let video = document.getElementById('video');
    let canvas = document.getElementById('canvas');
    let button = document.getElementById('button');

    let stopped = false;

    video.setAttribute('width', width);
    video.setAttribute('height', height);
    canvas.setAttribute('width', width);
    canvas.setAttribute('height', height);

    function takepicture() {
      if (stopped) {
        return;
      }
      let context = canvas.getContext('2d');
      canvas.width = width;
      canvas.height = height;
      context.drawImage(video, 0, 0, width, height);

      var data = canvas.toDataURL('image/png');
      sendValue(data);
    }


    function stopVideo() {
      video.pause();
      video.srcObject.getTracks()[0].stop();
      stopped = true;
    }

    function startVideo() {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
          video.srcObject = stream;
          video.play();
        })
        .catch(function(err) {
          console.log("An error occurred: " + err);
        });
    }

    function toggleVideo() {
      if (stopped) {
        startVideo();
        stopped = false;
      } else {
        stopVideo();
        stopped = true;
      }
      // Toggle the button text
      button.textContent = stopped ? startLabel : stopLabel;
    }

    if (navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then(function (stream) {
          video.srcObject = stream;
        })
        .catch(function (error) {
          console.log("Something went wrong!");
          console.error(error);
        });
    }

    button.addEventListener('click', toggleVideo);
    button.textContent = stopped ? startLabel : stopLabel;

    takepicture();
    setInterval(takepicture, debounce);
    window.rendered = true
  }
}

// Render the component whenever python send a "render event"
Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender)
// Tell Streamlit that the component is ready to receive events
Streamlit.setComponentReady()
// Don't actually need to display anything, so set the height to 0
Streamlit.setFrameHeight(0)
