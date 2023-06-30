const player = document.getElementById("video-player");
const progress = document.getElementById("progress_bar");
const progressBar = document.getElementById("progress_bar_filled");

var xhr = new XMLHttpRequest();
var xhr_styles = new XMLHttpRequest();
let count = false;

function submit() {
    if (count == false) {
        document.getElementById("description-container").style.display = "none";
        player.style.display = "block";
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
        "start_time_sec": startTime,
        "end_time_sec": endTime,
        "min_zoom": minZoom,
        "max_zoom": maxZoom,
        "min_angle": minAngle,
        "max_angle": maxAngle
    }
    try {
        fetch('/submit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(config)
        }).then(() => {
            getVideo();
        }).catch((error) => {
            console.log(error);
        });
    } catch (error) {
        console.log(error);
    }
};

function getVideo() {
    fetch('/get_video').then((response, error) => {    
        if (response.status === 200) {
            return response.blob();
        } else {
            throw new Error('Something went wrong on api server!\n', error);
        }
    }).then(function (videoBlob) {
        
        if (document.getElementById("video-container") != undefined) {
            let old_video = document.getElementById("video-container");
            player.removeChild(old_video);
            let new_video = document.createElement("video");
            new_video.id = "video-container";
            player.appendChild(new_video);
        } else {
            let video = document.createElement("video");
            video.id = "video-container";
            player.appendChild(video);
        }
        
        let video = document.getElementById("video-container");
        video.style.backgroundColor = "black";
        video.style.zIndex = "0";
        video.addEventListener("click", togglePlay);
        video.addEventListener("play", updatePlayBtn);
        video.addEventListener("pause", updatePlayBtn);
        video.addEventListener("timeupdate", handleProgress);
        
        let videoURL = URL.createObjectURL(videoBlob);
        let source = document.createElement("source");
        source.classList.add("source");
        source.src = videoURL;
        source.type = "video/mp4";
        video.appendChild(source);

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

function togglePlay() {
    let videoContainer = document.getElementById("video-container");
    if (videoContainer.paused || videoContainer.ended) {
        videoContainer.play();
    } else {
        videoContainer.pause();
    }
}

function updatePlayBtn() {
    let videoContainer = document.getElementById("video-container");
    playPauseBtn.innerHTML = videoContainer.paused ? "►" : "❚❚";
}

function handleProgress() {
    let videoContainer = document.getElementById("video-container");
    const progressPercentage = (videoContainer.currentTime / videoContainer.duration) * 100;
    progressBar.style.flexBasis = `${progressPercentage}%`;
}

function jump(e) {
    let videoContainer = document.getElementById("video-container");
    const position = (e.offsetX / progress.offsetWidth) * videoContainer.duration;
    videoContainer.currentTime = position;
}

const playPauseBtn = document.querySelector(".playPauseBtn");
playPauseBtn.addEventListener("click", togglePlay);

const videoContainer = document.getElementById("video-container");
videoContainer.addEventListener("click", togglePlay);
videoContainer.addEventListener("play", updatePlayBtn);
videoContainer.addEventListener("pause", updatePlayBtn);
videoContainer.addEventListener("timeupdate", handleProgress);

let mousedown = false;
progress.addEventListener("click", jump);
progress.addEventListener("mousedown", () => (mousedown = true));
progress.addEventListener("mousemove", (e) => mousedown && jump(e));
progress.addEventListener("mouseup", () => (mousedown = false));