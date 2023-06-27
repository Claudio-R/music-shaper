const videoContainer = document.querySelector(".video-container");
const playPauseBtn = document.querySelector(".playPauseBtn");
const progress = document.querySelector(".progress");
const player = document.querySelector(".player");
var xhr = new XMLHttpRequest();
var xhr_styles = new XMLHttpRequest();
let choises = [];
let count = false;

function storeArtistSong() {
    bottomMenuChoice = document.getElementById("artist").value;
    topMenuChoice = document.getElementById("song").value;

    choises = [bottomMenuChoice, topMenuChoice];
    
    xhr.open('POST', '/process_artist_song');
    xhr.setRequestHeader('Content-Type', 'application/json');

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

function storeStyleContent() { 
    style1_choice = document.getElementsByClassName('select-selected')[0].innerHTML;
    style2_choice = document.getElementsByClassName('select-selected')[1].innerHTML;
    content_choice = document.getElementsByClassName('select-selected')[2].innerHTML;

    style_content = [style1_choice, style2_choice, content_choice];
  
    xhr_styles.open('POST', '/process_style_content');
    xhr_styles.setRequestHeader('Content-Type', 'application/json');

    xhr_styles.onload = function() {
        if (xhr_styles.status === 200) {
            var response = JSON.parse(xhr_styles.responseText);
            console.log(response.message);
        }
    };

    xhr_styles.send(JSON.stringify({ "arrayDataChoices": style_content }));
}

function runClipGeneration() {
    console.log("sono entrato in eseguiScript()");
        try {
        const response = fetch('/execute_script', {
            method: 'POST',
            headers: {
            'Content-Type': 'application/json',
            },
            body: JSON.stringify({}),
        });
    
        if (response.status === 200) {
            const result = response.json();
        } else {
            console.error('Error in executing script');
        }
    
        printResponse(response);
    
        } catch (error) {
        console.error('Errore:', error);
        }
  };

function getVideo() {
    fetch('/get_video').then(function (response) {    
        if (response.status === 200) {
            printResponse(response);
            return response.blob();
        }
        throw new Error('Error retrieving video');

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

    if (count == true) {
        let spinner = document.getElementsByClassName("spinner")[0];
        player.removeChild(spinner);
        count = false;
        }
  })
  .catch(function(error) {
    console.log(error);
  });
}