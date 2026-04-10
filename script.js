const interval = 2000;
const timer = document.getElementById('timer');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const resultBox = document.getElementById('result-box');
const confidenceBox = document.getElementById('confidence-box');
const pauseButton = document.getElementById('pause-button');
const continuarButton = document.getElementById('continuar');
const anuncio = document.getElementById('anuncio');
const container = document.getElementById('daWholeThing');
let isProcessing = false;
let isPaused = true;
        
continuarButton.addEventListener('click', (event) => {
    event.preventDefault();
    isPaused = !isPaused;
    anuncio.style.display = 'none';
    container.style.display = 'inline-block';
});

function clicked(toPause) {
    if(toPause) {
        isPaused = !isPaused;
        pauseButton.textContent = isPaused ? "▶️" : "⏸️";
    } else {
        resultBox.textContent = '';
        confidenceBox.textContent = '';
    }
}

navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } }).then(stream => { 
    video.srcObject = stream; 
    resultBox.innerText = ""; 
    let innterInterval = interval;
    setInterval(() => {
        if (innterInterval >= 1000) {
            timer.innerHTML =`La próxima captura será tomada dentro de ${innterInterval/1000} segundos`;
            innterInterval -= 1000;
        } else {
            timer.innerHTML = 'Prediciendo...';
            captureAndPredict(); 
            innterInterval = interval;
        }
        if(isPaused) {
            timer.innerHTML = 'Procesamiento en pausa' 
        }  
    }, 1000);
})
    .catch(err => { 
        console.error("Error de cámara:", err); 
        resultBox.innerText = "No se pudo acceder a la cámara"; 
    });

function captureAndPredict() {
    if (isProcessing) return;
    isProcessing = true;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    tempCanvas.getContext('2d').drawImage(video, 0, 0);

    tempCanvas. toBlob(blob => {
        const formData = new FormData();
        formData.append('image', blob, 'capture.jpg');
        fetch('https://dactilologiamexicana.onrender.com/predict', {
        // fetch('http://127.0.0.1:10000/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if(data.error) {
                resultBox.innerText = "Error: " + data.error;
            } else {
                if(data.class == '_' && !isPaused) clicked(false);
                else if(data.class == '!') clicked(true);
                else if(!isPaused && data.class === 'None') confidenceBox.innerText = data.confidence
                else if(!isPaused) {
                    resultBox.innerHTML += data.class.toUpperCase();
                    confidenceBox.innerHTML = `Confianza: ${data.confidence}`;
                }       
            }
        })
        .catch(err => {
            console.error("Error:", err);
            resultBox.innerText = "Error con el servidor";
        })
        .finally(() => {
            isProcessing = false;
        });
    }, 'image/jpeg', 0.6);
}