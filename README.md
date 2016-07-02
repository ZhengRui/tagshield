## Server

Build:
```
g++ server/main.cpp -o server/visprivacyserv `pkg-config --cflags --libs opencv`
```

Usage:
```
./server/visprivacyserv config_file
```

A sample config file is `server/config`, in this file you can specify:

- 2 modes: `server mode`, `local mode`
- 3 tasks: `face detection`, `fist detection`, `marker detection`
- 4 types: `single image`, `folder contains many images`, `singe video`, `camera` in `local mode`

## Client Apps

- visPrivacy: either send cam-stream to server for image processing or do face/(single)marker detection on Android

*performance of opencv fist detection (haar-cascade palm detect) is really bad, a much better hand/gesture detector trained using __[faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)__ can be found [here](tbc)*
