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

function clickListener(event) {
  const {offsetX, offsetY} = event
  sendValue({x: offsetX, y: offsetY})
}

function onRender(event) {
  let {src, height, width} = event.detail.args

  const img = document.getElementById("image")

  if (img.src !== src || (height && img.height !== height) || (width && img.width !== width)) {
    img.src = src

    img.onload = () => {
      if (!width && !height) {
        width = img.naturalWidth
        height = img.naturalHeight
      }
      else if (!height) {
        height = width * img.naturalHeight / img.naturalWidth
      }
      else if (!width) {
        width = height * img.naturalWidth / img.naturalHeight
      }

      img.width = width
      img.height = height

      Streamlit.setFrameHeight(height + 10)

      // When image is clicked, send the coordinates to Python through sendValue
      if (!img.onclick) {
        img.onclick = clickListener
      }
    }
  }
}

// Render the component whenever python send a "render event"
Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender)
// Tell Streamlit that the component is ready to receive events
Streamlit.setComponentReady()
