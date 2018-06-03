# Face recognition using Tensorflow made easier and faster with a precached model exposed by a Python RPC server for use with Node.js

Reduces time to receiving a classification from around 11 seconds to < 100 ms by preloading trained model and only loading in the jpg for classification at run-time to feed through the preloaded network in Python. Also exposes an RPC server so you can call this Python function from any language you like such as Node.js which is much more common for real world use case on a web server at scale (eg web sockets and such).


## Background
The python code is heavily inspired by the original precict.py code in the directory above this one by the original author. However that file was not useful for general real world useage as it took about 11 seconds to load up the model each time and perform a classification which is less than ideal.

I decided to rewrite this to be optimised for web servers such as those running on Google Compute Engine allowing you to query faces super fast as you get the images from a client to classify in near real time (<100ms) using an remote procedure call to a pre cached model in python by refactoring the original code.


## Requirements and notes

### Python

Please ensure you change the commented variables near the top of the file to point to files on your system. Namely these are:

```python
MODEL = "/path/to/model.pb"
CLASSIFIER_FILENAME = "/path/to/your/classifier.pkl"
preload_image = "/path/to/some/image.jpg"
```
The preload_image is needed with some dummy jpg with a face to set the system up once before you make any RPC calls to the Python RPC server.


### Node.js

Ensure you change the following variable to a JPG you want to classify:

```javascript
var JPG_FILE = '/path/to/file.jpg';
```
Assuming the Python RPC server is running already and has printed out "Initiation complete" you can now use the example code to make RPC calls to Python from Node.js and it should be pretty fast - I am getting way under 100ms for a classification on a vanilla CPU instance on Google Compute Engine.

Please note that to run this you need to install zerorpc for node - which may need you to compile from source as I had issues using the NPM version. Once that is installed though all should work fine :-)

## Questions?

Feel free to [get in touch with Jason Mayes](http://www.jasonmayes.com) if any questions on setting this up in a production environment.
