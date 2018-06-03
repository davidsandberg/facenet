var zerorpc = require("zerorpc");

// Change this to a JPG to classify on your server.
var JPG_FILE = '/path/to/file.jpg';

// Connect to Python RPC server.
var client = new zerorpc.Client();
client.connect("tcp://127.0.0.1:4242");

// Invoke remote function call to classify a jpg file of our choice.
client.invoke("classifyFile", JPG_FILE, function(error, res, more) {
  // Print out classification.
  console.log(res.toString('utf8'));
});