$ nohup ~/.dropbox-dist/dropboxd&
$ screen
> streamer -c /dev/video1 -t 10 -r 1 -o 0000.jpeg &
> streamer -c /dev/video1 -t 300 -r 0.0033 -s 640x480 -o 0000.jpeg &

Dropbox:
Installed headless version.
Must keep running dropboxd process to keep folder syncing on. Hence using nohup to keep it going as a background process. There is a python file to itnerface with the headless dropbox; I haven't investigated it yet.

Explanation:
Run this inside the ~/Dropbox folder.

$ screen to keep streamer persistent. Does not seem to work with nohup, possibly because nohup supresses output or something.
Basic commands: 
screen -r (reattach)
screen -d (inside screen session, to detach)
ctrl-shift-w

ctrl-a w (list windows)
ctrl-a 1 (switch to window #1)
ctrl-a c (new window)
exit (close window) -- after last window is closed, screen terminates.


Streamer options:
Using the "second" camera, since this desktop has a built-in webcam mapped to /dev/video1
10 frames total
1 frame a second (rate)
Name starts at 0000.jpeg (it will automatically count up)


To handle background process:
$ ps x | grep streamer
$ sudo killall streamer

e.g. Replace "screamer" with "screen"

==========
*** streamer example
==========
This README.txt written by nouyang on 7 March 2018.
Details a remote monitoring setup.
Set a webcam to take a picture every 5 minutes, and then sync the pictures online via dropbox. Use the "share folder using link" option on dropbox to share with multiple people.

==========
*** streamer example
==========
streamer -c /dev/video1 -t 0:30 -o movie.avi -f jpeg 



==========
*** misc. itnernet notes
==========

DOES NOT WORK !!!!!!!!!$ nohup streamer -c /dev/video1 -t 300 -r 0.0033 -o 0000.jpeg&
$ nohup: ignoring input and appending output to `nohup.out'

http://www.netinstructions.com/automating-picture-capture-using-webcams-on-linuxubuntu/
o
# To take a picture every minute
# */1 * * * * streamer -f jpeg -s 1024x768 -o /home/stephen/timelap/$(date +\%m\%d\%k\%M).jpeg

# To take a picture every hour on the 15 minute mark using a different tool
# 15 * * * * fswebcam -r 1024x768 --jpeg 85 -D 4 -F 10 /home/stephen/webcamphotos/$(date +\%Y\%m\%d\%k\%M).jpeg

# Take a picture and upload it to the webserver every hour
@hourly bash /home/stephen/scripts/take_photo_and_push.sh
@e/stephen/scripts/take_photo_and_push.sh

==========
***
==========
oup vote
29
down vote
https://askubuntu.com/questions/106770/take-a-picture-from-terminal
If you're looking for something automated webcam is pretty decent. It has lots of lovely options for pushing the photos over the Internet.

If you want something more manual, and we're talking about a camera supported by V4L/UVC (most of them) you can use streamer to capture a frame from the device:

streamer -f jpeg -o image.jpeg



==========
***
==========
https://gist.github.com/lucasmezencio/3479301
streamer -o 0000.jpeg -s 300x200 -j 100 -t 2000 -r 1

==========
***
==========

streamer -c /dev/video1 -t 0:30 -o movie.avi -f jpeg 


streamer -h

==========
***
==========

 ~/.dropbox-dist/dropbox


