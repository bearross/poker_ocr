python version
python3.6 only

To install python packages, run follow command.
pip install -r requirements.txt

To test from image file.
python tester_image.py -n 2 -f test/2/1.jpg

Here, -n is the number of seat, -f is test file path.

To test from live game screen.
python tester_live.py -n 2 
Here,  -n is the number of seat.


To test carnum engine.
python tester_cardnum.py -f test/2/0.jpg

To run UI demo.
python run.py


To test from video file.
python tester_video_one.py -n 2 -f test/test.mp4
Here, -n is the number of seat, -f is test file path.

To test from multivideo file.
python tester_video_multi_run.py -f test/test.mp4
Here, -f is test file path.


To build exe file.
pyinstaller --noconsole --onefile run.py


