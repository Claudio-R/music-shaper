const videoContainer = document.querySelector(".video-container");
const playPauseBtn = document.querySelector(".playPauseBtn");
const progress = document.querySelector(".progress");
const progressBar = document.querySelector(".progress__filled");
const player = document.querySelector(".player");
var xhr = new XMLHttpRequest();
var xhr_styles = new XMLHttpRequest();
let choises = [];
let count = false;
const submitButton = document.getElementById('submit-button');


function storeNames() {
  //scelte artista e nome brano
  bottomMenuChoice = document.getElementById("artistName").value;
  topMenuChoice = document.getElementById("songName").value;

  choises = [bottomMenuChoice, topMenuChoice]; //salvo in un array nome e brano
  
  xhr.open('POST', '/process_array');
  xhr.setRequestHeader('Content-Type', 'application/json');

  // invio nome artista e brano
  xhr.onload = function () {
    if (xhr.status === 200) {
      var response = JSON.parse(xhr.responseText);
      console.log(response.message);
    }
  };

  xhr.send(JSON.stringify({ "arrayData": choises }));

  if (count == false) {
    let spinner = document.createElement("div");
    spinner.classList.add("spinner");
    player.appendChild(spinner);
    count = true;
  }

}

function storeStyles() { 
  //prendo le scelte di stile
  style1_choice = document.getElementsByClassName('select-selected')[0].innerHTML;
  style2_choice = document.getElementsByClassName('select-selected')[1].innerHTML;
  content_choice = document.getElementsByClassName('select-selected')[2].innerHTML;

  style_choices = [style1_choice, style2_choice, content_choice];

  
  //invio gli stili
  xhr_styles.open('POST', '/process_array_choices');
    xhr_styles.setRequestHeader('Content-Type', 'application/json');

    xhr_styles.onload = function() {
      if (xhr_styles.status === 200) {
       var response = JSON.parse(xhr_styles.responseText);
        console.log(response.message);
      }
    };

    xhr_styles.send(JSON.stringify({ "arrayDataChoices": style_choices }));
}


// RICEVO IL VIDEO
function getVideo() {
  console.log("sono entrato in getVideo()");
    fetch('/video').then(function (response) {    
      if (response.ok) {
        return response.blob();
      }
      throw new Error('Errore nella richiesta del video');

    }).then(function (videoBlob) {
    var videoURL = URL.createObjectURL(videoBlob);

    var source_1 = document.createElement("source");
    source_1.classList.add("source");
    videoContainer.appendChild(source_1);
    source_1.src = videoURL;
    source_1.type = "video/mp4";

    var source_2 = document.createElement("source");
    source_2.classList.add("source");
    videoContainer.appendChild(source_2);
    source_2.src = videoURL;
    source_2.type = "video/webm";

    document.getElementsByClassName("spinner")[0].remove();    
  })
  .catch(function(error) {
    console.log(error);
  });
}


function togglePlay() {
  if (videoContainer.paused || videoContainer.ended) {
    videoContainer.play();
  } else {
    videoContainer.pause();
  }
}

function updatePlayBtn() {
  playPauseBtn.innerHTML = videoContainer.paused ? "►" : "❚❚";
}

function handleProgress() {
  const progressPercentage = (videoContainer.currentTime / videoContainer.duration) * 100;
  progressBar.style.flexBasis = `${progressPercentage}%`;
}

function jump(e) {
  const position = (e.offsetX / progress.offsetWidth) * videoContainer.duration;
  videoContainer.currentTime = position;
}

function myFunction() {
  document.getElementById("myDropdown").classList.toggle("show");
}

window.onclick = function(event) {
  if (!event.target.matches('.dropbtn')) {
    var dropdowns = document.getElementsByClassName("dropdown-content");
    var i;
    for (i = 0; i < dropdowns.length; i++) {
      var openDropdown = dropdowns[i];
      if (openDropdown.classList.contains('show')) {
        openDropdown.classList.remove('show');
      }
    }
  }
}

playPauseBtn.addEventListener("click", togglePlay);
videoContainer.addEventListener("click", togglePlay);
videoContainer.addEventListener("play", updatePlayBtn);
videoContainer.addEventListener("pause", updatePlayBtn);
videoContainer.addEventListener("timeupdate", handleProgress);

let mousedown = false;
progress.addEventListener("click", jump);
progress.addEventListener("mousedown", () => (mousedown = true));
progress.addEventListener("mousemove", (e) => mousedown && jump(e));
progress.addEventListener("mouseup", () => (mousedown = false));

function eseguiScript() {
  console.log("sono entrato in eseguiScript()");
  try {
    const response = fetch('/execute_script', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({}),
    });

    if (response.ok) {
      const result = response.json();
    } else {
      console.error('Errore nella richiesta:', response.status);
    }
  } catch (error) {
    console.error('Errore:', error);
  }
};




