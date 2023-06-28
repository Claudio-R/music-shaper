const videoContainer = document.querySelector(".video-container");
const player = document.querySelector(".player");
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

    config = {
        "artist": artist,
        "song": song,
        "style1": style1,
        "style2": style2,
        "content": content
    }
    
    fetch('/submit', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(config)
    }).then((response, error) => {
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