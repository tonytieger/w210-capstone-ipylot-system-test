## Model Creation:

Inside the "/Drowsiness/Model" folder our model called "BlinkModel.t7" already exists, which is the one that came from DBSE-monitor github project (https://github.com/altaga/DBSE-monitor/tree/master/Drowsiness/Model)". The one we trained is currently being used (with our own eyes added to the database).

The training has the following parameters as input.

- input image shape: (24, 24)
- validation_ratio: 0.1
- batch_size: 64
- epochs: 40
- learning rate: 0.001
- loss function: cross entropy loss
- optimizer: Adam

In the first part of the code you can modify the parameters, according to your training criteria.

# How to run
In a jetsan, run "build.sh" to build the dockerfile. The dockerfile will create a new user, and this user will be set with audio so that the docker container's audio will use the Jetsan's audio. We use the latest (as of 12/2021) NVidia ML container for Jetscan that has pytorch, jupyter, etc.

Then run "host_runner.sh" to start the container with the proper options to allow webcam access and audio access.

Then, you can run "jupyter notebook --ip 0.0.0.0" and access the notebook from a browser. After running the notebook, if changes are made, you may need to rerun it more than once, as the first rerun will result in a failure since the camera resources are not properly releated (the code just runs in an infinite loop, reading from the webcam).

Website:
https://www.ipylot.com

Paper:
https://www.ipylot.com/paper
