const videoContainer = document.getElementById("video-container");
const playPauseBtn = document.querySelector(".playPauseBtn");
const progress = document.getElementById("progress_bar");
const progressBar = document.getElementById("progress_bar_filled");

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