### Utils per i video
- estrazione di frame dal video: `ffmpeg -i 192.168.0.74_01_20190717024535302_2.mp4 -r 1 -q:v 1 -qmin 1 -qmax 1 frames/video%04d.jpg`
- rebuild del video a partire dai frames: `ffmpeg -framerate 25 -i frames/video%04d.jpg annotated_video.mp4`