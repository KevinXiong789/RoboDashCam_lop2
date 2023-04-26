# RoboDashCam_lop2
Here is just a changed code *lop_node.py* in RoboDashCam project.
## How to replace
1. First you need to have the complete project file of RoboDashCam
2. then create a Container according to the README in lop2
3. enter the Container and find the *lop_node.py*
4. replace it with the code here
5. In Container terminal, run: `pip3 install influxdb-client`, then try `python3`,then `from influxdb_client import InfluxDBClient`, if no error, good, if error, try `pip3 install --upgrade requests`, and start new terminal
## Run
* `colcon build --symlink-install --packages-select lop2`
* `. ./install/setup.bash`
* `ros2 launch lop2 main.launch.py`
## Finished Task
* Use OpenPose get xyz coordinate of keypoints and send them to InfluxDB, here in code use right hand and neck as example
* InfluxDB install: https://portal.influxdata.com/downloads/
