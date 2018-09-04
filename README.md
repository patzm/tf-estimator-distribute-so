# tf-estimator-distribute-so
This repository serves as a minimal, reproducible example to the issue I describe in 
[this SO issue](https://stackoverflow.com/questions/50924287/how-to-influence-the-name-scope-of-collated-variables-slots)
.

## Setup
```
git clone https://github.com/patzm/tf-estimator-distribute-so.git
cd tf-estimator-distribute-so
pip install -r requirements.txt
```

## Run the example
```
python reproduce_read_x.py
```

## Launch tensorboard
To see the graph representation in TensorBoard, launch it with the `logdir` that is printed during the example
execution.
If executed with the `OneDeviceStrategy`, a lot of `Read_<num>` operators are added to the graph without a parent scope.
If executed with the `MirroredStrategy`, a lot of `group_deps_<num>` operators are added to the graph without a parent
scope.
The following two images show this behavior:
* `OneDeviceStrategy`:
  
  ![OneDeviceStrategy](https://raw.githubusercontent.com/patzm/tf-estimator-distribute-so/master/images/OneDeviceStrategy.png)

* `MirroredStrategy`:

  ![MirroredStrategy](https://raw.githubusercontent.com/patzm/tf-estimator-distribute-so/master/images/MirroredStrategy.png)

This makes the graph totally unreadable.
It would be great if all these variables were at least enclosed in a `tf.variable_scope`.
