
# Standart library:
import os
import math

# Installed library:
import cv2
import torch
import numpy as np

# *****************************************************************
# Before import run first: pip3 install influxdb-client
# If Error with influxdb, try this: pip3 install --upgrade requests
# InfluxDB library:
from influxdb_client import InfluxDBClient, Point
import time
# *****************************************************************

# ROS libraries:
import rclpy
import rclpy.node

import std_msgs.msg
from std_msgs.msg import Float32MultiArray
import sensor_msgs.msg
import visualization_msgs.msg
import geometry_msgs.msg
import cv_bridge
import image_geometry
import message_filters

import ament_index_python.packages

# Local libraries:
from .with_mobilenet import PoseEstimationWithMobileNet
from .keypoints import extract_keypoints, group_keypoints, BODY_PARTS_KPT_IDS
from .load_state import load_state
from .pose import Pose, track_poses
from .val import normalize, pad_width


NUMBER_OF_TEST_PIXELS = 5
REALSENSE_RESOLUTION = (1280, 720)
TEST_PIXELS = np.full(shape=(Pose.number_of_keypoints, 2), fill_value=(REALSENSE_RESOLUTION[1] // 2), dtype=int)

for index in range( (NUMBER_OF_TEST_PIXELS + 1) ):
	TEST_PIXELS[index][1] = ( (REALSENSE_RESOLUTION[0] // (NUMBER_OF_TEST_PIXELS + 1) ) * (index + 1) )


# OpenCV uses BGR-order:
OPENCV_COLORS = {
	'red':     (  0,   0, 255),
	'green':   (  0, 255,   0),
	'blue':    (255,   0,   0),
	'cyan':    (255, 255,   0),
	'magenta': (255,   0, 255),
	'yellow':  (  0, 255, 255)
}

ROS_COLORS = {
	'red':     std_msgs.msg.ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
	'green':   std_msgs.msg.ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
	'blue':    std_msgs.msg.ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
	'cyan':    std_msgs.msg.ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0),
	'magenta': std_msgs.msg.ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0),
	'yellow':  std_msgs.msg.ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
}

MARKER_SIZE = geometry_msgs.msg.Vector3(x=0.1, y=0.1, z=0.1)
MIN_KEYPOINT_DISTANCE_TO_CAMERA = 0.05
USE_CPU = False

# ******************************************************************************
#set necessary info to connect InfluxDBClient
influxdb_client = InfluxDBClient(url="http://localhost:8086", \
								token="EZZyYWflA8jgJFT1J5TfTkTbgECQQzcIbEXvTDKwBVKntwRm4JyAEy3wzjzJE20i-i-8k9vFbIO1WDxsGNQSPw==", \
								org="PointCloud")
influxdb_write_api = influxdb_client.write_api()
# ******************************************************************************

class Pose3D ():

	def __init__ (self, confidence=None) -> None:
		self.points = list()

		self.confidence = -1 if confidence is None else confidence



class Lop (rclpy.node.Node):

	def __init__ (self):
		super().__init__(node_name='lop_node')
		#self.get_logger().info('LOP initialising.')


		# Parameters:
		self.declare_parameter(name='publish_images', value=False)
		self.declare_parameter(name='publish_markers', value=False)

		self.publish_images =  (self.get_parameter(name='publish_images').value)  # .lower() ==  'true')
		self.publish_markers = (self.get_parameter(name='publish_markers').value)  # .lower() == 'true')


		# CV bridge:
		self.bridge = cv_bridge.CvBridge()


		# Message synchronisation:
		image_subscriber = message_filters.Subscriber(self, sensor_msgs.msg.Image,
													  '/camera/color/image_raw')
		info_subscriber = message_filters.Subscriber(self, sensor_msgs.msg.CameraInfo,
													 '/camera/color/camera_info')

		if self.publish_markers:
			depth_subscriber = message_filters.Subscriber(self, sensor_msgs.msg.Image,
														  '/camera/aligned_depth_to_color/image_raw')

		subscribers = (image_subscriber, info_subscriber, depth_subscriber) \
						if self.publish_markers else (image_subscriber, info_subscriber)

		time_synchronizer = message_filters.TimeSynchronizer(fs=subscribers, queue_size=10)
		time_synchronizer.registerCallback(self.callback)


		# Publishers:
		if self.publish_images:
			self.image_publisher = self.create_publisher(topic='/new_camera/image_raw',
														 msg_type=sensor_msgs.msg.Image, qos_profile=10)
			self.image_info_publisher = self.create_publisher(topic='/new_camera/camera_info',
															  msg_type=sensor_msgs.msg.CameraInfo,
															  qos_profile=10)

		if self.publish_markers:
			self.marker_publisher = self.create_publisher(topic='/new_camera/markers',
														  msg_type=visualization_msgs.msg.MarkerArray,
														  qos_profile=10)

		# ************************************************************************************************
		# Publisher for handposition
		self.handposition_publisher = self.create_publisher(Float32MultiArray, '/Openpose/hand_position', 10)
		# ************************************************************************************************


		# Neural network setup:
		self.net = PoseEstimationWithMobileNet()

		share_directory_path_string = ament_index_python.packages.get_package_share_directory(package_name='lop2')
		checkpoint_file_path = os.path.join(share_directory_path_string, 'data', 'checkpoint_iter_370000.pth')
		checkpoint = torch.load(f=checkpoint_file_path, map_location='cpu')
		load_state(self.net, checkpoint)

		self.net = self.net.eval()

		if not USE_CPU:
			self.net = self.net.cuda()

		self.stride = 8
		self.upsample_ratio = 4
		self.previous_poses = []
		self.height_size = 256
		self.track = True
		self.smooth = True


	def callback (self, camera_message, camera_info_message, depth_message=None) -> None:
		#poses = (Pose(keypoints=TEST_PIXELS, confidence=90.0), )

		cv_image = self.bridge.imgmsg_to_cv2(img_msg=camera_message, desired_encoding='bgr8')
		#cv_image = self.lop(image=cv_image)

		poses = self.lop(image=cv_image)
		#self.get_logger().info(f'Number of poses: {len(poses) }')

		if self.publish_images:
			self.draw_keypoints(image=cv_image, poses=poses)
			image_message = self.bridge.cv2_to_imgmsg(cvim=cv_image, encoding='bgr8',
													header= camera_info_message.header)
			self.image_publisher.publish(image_message)

			self.image_info_publisher.publish(camera_info_message)

		if self.publish_markers:
			self.calculate_depth(depth_message, camera_info_message, poses)


	def draw_keypoints (self, image, poses):
		for pose in poses:
			#self.get_logger().info(f'Pose [Confidence: {pose.confidence}]\n: {pose.keypoints}')

			for point in pose.keypoints:
				if any( (coordinate < 0) for coordinate in point):
					continue

				cv2.circle(img=image, center=tuple(point), radius=15, color=OPENCV_COLORS['blue'], thickness=-1)


			if self.publish_markers:  # Markers replace lines in image.
				continue

			for keypoint_a_index, keypoint_b_index in BODY_PARTS_KPT_IDS[:-2]:

				if any( any( (coordinate < 0) for coordinate in pose.keypoints[index] )
						for index in (keypoint_a_index, keypoint_b_index) ):
					continue

				cv2.line(img=image, pt1=tuple(pose.keypoints[keypoint_a_index] ), pt2=tuple(pose.keypoints[keypoint_b_index] ),
						 color=OPENCV_COLORS['yellow'], thickness=10)


	def calculate_depth (self, depth_message, camera_info_message, poses) -> None:

		depth_image = self.bridge.imgmsg_to_cv2(img_msg=depth_message, desired_encoding='16UC1')

		pinhole_camera_model = image_geometry.PinholeCameraModel()
		pinhole_camera_model.fromCameraInfo(msg=camera_info_message)

		poses_3d = list()

		for pose_idx, pose in enumerate(poses):
			pose_3d = Pose3D(confidence=pose.confidence)

			for pixel_coordinates in pose.keypoints:

				if any( (coordinate < 0) for coordinate in pixel_coordinates):  # No keypoint detected:
					pose_3d.points.append(None)
					continue

				unit_vector_3d = pinhole_camera_model.projectPixelTo3dRay(uv=pixel_coordinates)
				depth = (depth_image[tuple(pixel_coordinates[::-1] ) ] / 1000)  # Millimeter to meter.

				point_3d = tuple( (element * depth) for element in unit_vector_3d)
				pose_3d.points.append(point_3d)

			poses_3d.append(pose_3d)

			
			# ***********InfluxDB***************************
			# keypoint_ID, Neck=1, right hand=4, details Info see here https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_02_output.html
			keypoint_ID = 1
			point_name = f"Pose{pose_idx}_Neck_Openpose"
			people_ID = pose_idx + 1
			#keypoint_ID = 4
			#point_name = "right_hand_Openpose"
			
			if (pose_3d.points[keypoint_ID] is None) or (pose_3d.points[keypoint_ID][0]==0 and pose_3d.points[keypoint_ID][1]==0 and pose_3d.points[keypoint_ID][2]==0):
				pass
				
			else:
				distance_to_zero = math.sqrt((pose_3d.points[keypoint_ID][0]**2)+(pose_3d.points[keypoint_ID][1]**2)+(pose_3d.points[keypoint_ID][2]**2))
				timestamp = int(time.time()*1000)
				data_point = Point(point_name) \
							.field("x", pose_3d.points[keypoint_ID][0]) \
							.field("y", pose_3d.points[keypoint_ID][1]) \
							.field("d", pose_3d.points[keypoint_ID][2]) \
							.field("distance", distance_to_zero) \
							.field("people_ID", people_ID) \
							.time(timestamp,"ms")
				influxdb_write_api.write(bucket="OpenPose_test", record=data_point)
			
			# *******************************************
			
			# **********get right wrist and elbow position and publish***********************************
			if (pose_3d.points[4] is not None) and (pose_3d.points[5] is not None):
				msg = Float32MultiArray()
				msg.data = [pose_3d.points[4][0], pose_3d.points[4][1], pose_3d.points[4][2], pose_3d.points[5][0], pose_3d.points[5][1], pose_3d.points[5][2]]
				self.handposition_publisher.publish(msg)
			'''
			else:
				msg = Float32MultiArray()
				msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
				self.handposition_publisher.publish(msg)
			'''
			# *******************************************************************************************
			



		self.create_pose_markers(poses_3d)


	@staticmethod
	def validate_point (point) -> float:

		if point is None:
			return False


		distance = math.sqrt( sum( (coordinate * coordinate) for coordinate in point) )

		return (distance > MIN_KEYPOINT_DISTANCE_TO_CAMERA)


	def create_pose_markers (self, poses) -> None:
		# visualization_msgs.msg.Marker.LINE_LIST only uses the x-component, for line width:
		marker_scale = geometry_msgs.msg.Vector3(x=0.01)

		marker_array = visualization_msgs.msg.MarkerArray()

		time_stamp = self.get_clock().now().to_msg()

		for person_index, pose in enumerate(poses):
			# Exclude bodyparts ear-to-shoulder, the last two:
			# Double list-comprehension to flatten the result:
			marker_points = tuple(points for keypoint_a_index, keypoint_b_index in BODY_PARTS_KPT_IDS[:-2]
								  for points in (pose.points[keypoint_a_index], pose.points[keypoint_b_index] )
								  if all(self.validate_point(pose.points[index] ) for index in (keypoint_a_index, keypoint_b_index) ) )

			marker_color = ROS_COLORS[tuple(ROS_COLORS.keys() ) [person_index % len(ROS_COLORS) ] ]

			marker_array.markers.append(self.create_marker(index=person_index, points=marker_points, time_stamp=time_stamp,
														   color=marker_color, scale=marker_scale,
														   type=visualization_msgs.msg.Marker.LINE_LIST) )

		self.marker_publisher.publish(marker_array)


	def create_marker (self, index, points, time_stamp, type=visualization_msgs.msg.Marker.POINTS,
					   color=ROS_COLORS['red'], scale=MARKER_SIZE) -> visualization_msgs.msg.Marker:

		marker = visualization_msgs.msg.Marker()

		marker.id = index
		marker.ns = 'lop'
		marker.color = color
		marker.action = visualization_msgs.msg.Marker.ADD
		marker.type = type
		marker.scale = scale
		marker.pose.orientation.w = 1.0
		marker.header.stamp = time_stamp
		marker.header.frame_id = 'camera_color_optical_frame'
		marker.lifetime.sec = 1  # Seconds.
		marker.points = tuple(geometry_msgs.msg.Point(x=point[0], y=point[1], z=point[2] )
							  for point in points)

		return marker


	def infer_fast (self, net, img, net_input_height_size, stride, upsample_ratio, cpu,
					pad_value=(0, 0, 0), img_mean=np.array( [128, 128, 128], np.float32),
					img_scale=np.float32(1/256) ):

		scale = net_input_height_size / img.shape[0]

		scaled_img = cv2.resize(src=img, dsize=(0, 0), fx=scale, fy=scale,
								interpolation=cv2.INTER_LINEAR)
		scaled_img = normalize(scaled_img, img_mean, img_scale)
		min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size) ]
		padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

		tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
		if not cpu:
			tensor_img = tensor_img.cuda()

		stages_output = net(tensor_img)

		stage2_heatmaps = stages_output[-2]
		heatmaps = np.transpose(a=stage2_heatmaps.squeeze().cpu().data.numpy(), axes=(1, 2, 0) )
		heatmaps = cv2.resize(src=heatmaps, dsize=(0, 0), fx=upsample_ratio, fy=upsample_ratio,
							  interpolation=cv2.INTER_CUBIC)

		stage2_pafs = stages_output[-1]
		pafs = np.transpose(a=stage2_pafs.squeeze().cpu().data.numpy(), axes=(1, 2, 0) )
		pafs = cv2.resize(src=pafs, dsize=(0, 0), fx=upsample_ratio, fy=upsample_ratio,
						  interpolation=cv2.INTER_CUBIC)

		return heatmaps, pafs, scale, pad


	def lop (self, image):

		heatmaps, pafs, scale, pad = self.infer_fast(self.net, image, self.height_size,
													 self.stride, self.upsample_ratio, USE_CPU)

		total_keypoints_num = 0
		all_keypoints_by_type = []
		for keypoint_idx in range(Pose.number_of_keypoints):  # 19th for bg
			total_keypoints_num += extract_keypoints(heatmaps[:, :, keypoint_idx], all_keypoints_by_type,
													 total_keypoints_num)

		pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)

		for keypoint_id in range(all_keypoints.shape[0] ):
			all_keypoints[keypoint_id, 0] = (all_keypoints[keypoint_id, 0] * self.stride / self.upsample_ratio - pad[1] ) / scale
			all_keypoints[keypoint_id, 1] = (all_keypoints[keypoint_id, 1] * self.stride / self.upsample_ratio - pad[0] ) / scale

		current_poses = []
		for pose_entry in pose_entries:

			if len(pose_entry) == 0:
				continue

			pose_keypoints = np.full(shape=(Pose.number_of_keypoints, 2), fill_value=-1, dtype=np.int32)

			for keypoint_id in range(Pose.number_of_keypoints):
				if pose_entry[keypoint_id] != -1.0:  # keypoint was found
					pose_keypoints[keypoint_id, 0] = int(all_keypoints[int(pose_entry[keypoint_id] ), 0] )
					pose_keypoints[keypoint_id, 1] = int(all_keypoints[int(pose_entry[keypoint_id] ), 1] )

			current_poses.append(Pose(pose_keypoints, pose_entry[18] ) )


		if self.track:
			track_poses(self.previous_poses, current_poses, smooth=self.smooth)
			self.previous_poses = current_poses

		return current_poses



def main (args=None):
	rclpy.init(args=args)

	image_repeater = Lop()
	rclpy.spin(image_repeater)
	image_repeater.destroy_node()

	rclpy.shutdown()



if __name__ == '__main__':
	main()

