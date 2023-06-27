//NOTE - VideoContainer has been already defined
// const videoContainer = document.querySelector(".video-container");
const playPauseBtn = document.querySelector(".playPauseBtn");
const progress = document.querySelector(".progress");
const progressBar = document.querySelector(".progress__filled");
const player = document.querySelector(".player");

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