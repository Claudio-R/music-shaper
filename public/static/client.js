const videoContainer = document.getElementById("video-container");
const player = document.getElementById("video-player");
var xhr = new XMLHttpRequest();
var xhr_styles = new XMLHttpRequest();
let choises = [];
let count = false;

function submit() {
    if (count == false) {
        let spinner = document.createElement("div");
        spinner.classList.add("spinner");
        player.appendChild(spinner);
        count = true;
    }

    artist = document.getElementById("artist").value;
    song = document.getElementById("song").value;
    style1 = document.getElementsByClassName('select-selected')[0].innerHTML;
    style2 = document.getElementsByClassName('select-selected')[1].innerHTML;
    content = document.getElementsByClassName('select-selected')[2].innerHTML;
    startTime = document.getElementById("startTime").value;
    endTime = document.getElementById("endTime").value;
    minZoom = document.getElementById("minZoom").value;
    maxZoom = document.getElementById("maxZoom").value;
    minAngle = document.getElementById("minAngle").value;
    maxAngle = document.getElementById("maxAngle").value;

    config = {
        "artist": artist,
        "song": song,
        "style1": style1,
        "style2": style2,
        "content": content,
        "startTime": startTime,
        "endTime": endTime,
        "minZoom": minZoom,
        "maxZoom": maxZoom,
        "minAngle": minAngle,
        "maxAngle": maxAngle
    }
    
    fetch('/submit', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(config)
    }).then((response, error) => {
        getVideo();
        if (response.status === 200) {
            console.log(response.message);
        } else {
            throw new Error('Something went wrong on api server!');
        }
    }
    ).catch((error) => {
        console.log(error);
    });
};

function getVideo() {
    fetch('/get_video').then((response, error) => {    
        if (response.status === 200) {
            return response.blob();
        } else {
            throw new Error('Something went wrong on api server!\n', error);
        }
    }).then(function (videoBlob) {
        const videoURL = URL.createObjectURL(videoBlob);
        document.getElementById("video").style.display = "block";
        let source;
        if (document.getElementsByClassName("source")[0] != undefined) {
            source = document.getElementsByClassName("source")[0];
        } else {
            source = document.createElement("source");
            source.classList.add("source");
            videoContainer.appendChild(source);
        }
        source.src = videoURL;
        source.type = "video/mp4";

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