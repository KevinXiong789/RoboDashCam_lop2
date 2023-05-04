//*********subscribe point cloud and right wrist and elbow position from cam *******************************************testing
//*********crop, vg sor filters, then get point cloud, get tool grasping position (xyz and angle), then pub*****************************
//*********also Hand in cell for 3s -> Handover triggered**************************************

#include <iostream>
#include <stdio.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/bool.hpp>
#include <math.h>
#include <angles/angles.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/angles.h>
#include <pcl/kdtree/kdtree_flann.h>


class SubscriberNode : public rclcpp::Node
{
public:
  SubscriberNode() : Node("Handover_detector")
  {
    // Subscribe to the point cloud topic
    pcl_subscriber_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "/camera/depth/color/points", 10, std::bind(&SubscriberNode::processPointCloud, this, std::placeholders::_1));

    // Subscribe to the Float32MultiArray topic
    float_array_subscriber_ = create_subscription<std_msgs::msg::Float32MultiArray>(
      "/Openpose/hand_position", 10, std::bind(&SubscriberNode::processFloatArray, this, std::placeholders::_1));

    // Create a publisher to publish the processed point cloud
    pcl_publisher = create_publisher<sensor_msgs::msg::PointCloud2>("/processed_point_cloud", 1);

    // Create a publisher to publish the tool grasp position and angle
    tool_grasping_pub = create_publisher<std_msgs::msg::Float32MultiArray>("/handover/tool_grasping_position", 1);

    // Create publisher for handover trigged Bool
    handover_bool_pub = create_publisher<std_msgs::msg::Bool>("/handover/handover_trigger", 1);

    prev_time = this->now();
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_subscriber_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr float_array_subscriber_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_publisher;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr tool_grasping_pub;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr handover_bool_pub;
  rclcpp::Time prev_time;

  // Position date will be used in all function
  float right_hand_x;
  float right_hand_y;
  float right_hand_z;
  float right_elbow_x;
  float right_elbow_y;
  float right_elbow_z;

  // Value for handover trigged
  float prev_x = 0;
  float prev_y = 0;
  float prev_z = 0;
  
  float timer_duration;
  float static_hand_duration = 3.0;
  std_msgs::msg::Bool handover_trigged_msg;


  float distanceComputing (Eigen::Vector4f point, Eigen::Vector4f point2){
    //compute the distance between 2 points
    float distance;
    distance= sqrt(pow(point[0]-point2[0],2)+pow(point[1]-point2[1],2)+pow(point[2]-point2[2],2));
    return distance;
  }

  bool is_in_the_cell(){
    if (right_hand_x >= -0.1 && right_hand_x <= 0.1 &&
        right_hand_y >= -0.1 && right_hand_y <= 0.1 &&
        right_hand_z >= 0.4 && right_hand_z <= 0.6){
          return true;
        }
    return false;
  }


  void processPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // Convert PointCloud2 to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *pcl_cloud);

    // Crop point cloud around right hand
    pcl::CropBox<pcl::PointXYZ> crop_filter;
    // Set the center of the crop box to the right hand
    crop_filter.setTranslation(Eigen::Vector3f(right_hand_x, right_hand_y, right_hand_z));
    crop_filter.setInputCloud(pcl_cloud);
    // Set the size of the crop box to 0.2 x 0.2 x 0.2
    Eigen::Vector4f min_point(-0.2, -0.2, -0.2, 1);
    Eigen::Vector4f max_point(0.2, 0.2, 0.2, 1);
    crop_filter.setMin(min_point);
    crop_filter.setMax(max_point);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    crop_filter.filter(*cropped_cloud);


    // Check if the cropped_cloud is empty before applying the VoxelGrid filter
    // with this VoxelGrid filter, FPS>29
    pcl::PointCloud<pcl::PointXYZ>::Ptr processed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (!cropped_cloud->empty()){
      // VoxelGrid Filter point cloud around right hand
      pcl::PointCloud<pcl::PointXYZ>::Ptr filter_1_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::VoxelGrid <pcl::PointXYZ> vg_filter;
      vg_filter.setInputCloud(cropped_cloud);
      vg_filter.setLeafSize(0.01f,0.01f,0.01f);
      vg_filter.filter(*filter_1_cloud);

      // add SOR filter, after vg filter,sor filter works good, FPS>29
      pcl::StatisticalOutlierRemoval <pcl::PointXYZ> SOR_filter;
      SOR_filter.setInputCloud(filter_1_cloud);
      SOR_filter.setMeanK(30);
      SOR_filter.setStddevMulThresh(1);
      SOR_filter.filter(*processed_cloud);

    } else {
      *processed_cloud = *cropped_cloud;
    }


    
    pcl::PointCloud<pcl::PointXYZ> processed_cloud_const = *processed_cloud;
    Eigen::Vector4f elbow_point = Eigen::Vector4f(right_elbow_x, right_elbow_y, right_elbow_z,1);
    Eigen::Vector4f tool_max_point;
    // get the max distance in processed cloud from elbow
    pcl::getMaxDistance(processed_cloud_const, elbow_point, tool_max_point);
    //RCLCPP_INFO(get_logger(), "tool_max_point xyz: (%.4f, %.4f, %.4f)", tool_max_point[0], tool_max_point[1], tool_max_point[2]);
    // Compute the center of mass of the cropped point cloud
    //Eigen::Vector4f hand_centroid;
    //pcl::compute3DCentroid(*processed_cloud, hand_centroid);
    //RCLCPP_INFO(get_logger(), "Center of mass: (%.4f, %.4f, %.4f)", hand_centroid[0], hand_centroid[1], hand_centroid[2]);
    Eigen::Vector4f wrist_point = Eigen::Vector4f(right_hand_x, right_hand_y, right_hand_z,1);

    float distance_threshold = 0.2;
    
    if (!isnan(tool_max_point[0]) && distanceComputing(tool_max_point, wrist_point)> distance_threshold){
      RCLCPP_INFO(get_logger(), "Tool in hand!!!!!!!!!!!!!");

      // search for the nearest points from max point to compute the centroid
      // and so to get the tool grasping position
      
      pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
      kdtree.setInputCloud (processed_cloud);
      pcl::PointXYZ searchPoint;
      searchPoint.x=tool_max_point[0];
      searchPoint.y=tool_max_point[1];
      searchPoint.z=tool_max_point[2];
      std::vector<int> pointIdxRadiusSearch;
      std::vector<float> pointRadiusSquaredDistance;
      float radius = 0.03;
      pcl::PointCloud<pcl::PointXYZ>::Ptr max_cloud (new pcl::PointCloud<pcl::PointXYZ>);
      if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
        {
          
          max_cloud->width = pointIdxRadiusSearch.size ();
          max_cloud->height = 1;
          max_cloud->points.resize (max_cloud->width * max_cloud->height);
          for (std::size_t i = 0; i < pointIdxRadiusSearch.size (); ++i) {
            pcl::PointXYZ point;
            max_cloud->points[i].x=processed_cloud->points[ pointIdxRadiusSearch[i] ].x;
            max_cloud->points[i].y=processed_cloud->points[ pointIdxRadiusSearch[i] ].y;
            max_cloud->points[i].z=processed_cloud->points[ pointIdxRadiusSearch[i] ].z;
            
          }
        }
      const pcl::PointCloud<pcl::PointXYZ> max_cloud_const=*max_cloud;
      Eigen::Vector4f toolgrasp_centroid;
      // tool grasping position is toolgrasp_centroid
      pcl::compute3DCentroid(max_cloud_const, toolgrasp_centroid);
      //printf("toolgrasp_position: %.4f, %.4f, %.4f \n",toolgrasp_centroid(0),toolgrasp_centroid(1),toolgrasp_centroid(2));


      // get the angle to grasp the tool
      // compute the tool vector
      const Eigen::Vector3f tool_vector = Eigen::Vector3f(toolgrasp_centroid(0)-right_hand_x, toolgrasp_centroid(1)-right_hand_y, toolgrasp_centroid(2)-right_hand_z);
      //unit vectors
      const Eigen::Vector3f x_vector = Eigen::Vector3f(1,0,0);
      const Eigen::Vector3f y_vector = Eigen::Vector3f(0,1,0);
      //const Eigen::Vector3f z_vector = Eigen::Vector3f(0,0,1);
      //for roll
      double R_angle = std::acos((tool_vector-tool_vector.dot(x_vector)*x_vector).normalized().dot(y_vector));
      //for pitch
      double P_angle = std::acos((tool_vector-tool_vector.dot(y_vector)*y_vector).normalized().dot(x_vector));

      float roll_command_deg = pcl::rad2deg(R_angle+angles::from_degrees(-180));
      float pitch_command_deg = pcl::rad2deg(-P_angle+angles::from_degrees(90));
      //printf("roll: %.4f, pitch: %.4f \n", roll_command_deg, pitch_command_deg);

      // publish tool grasping position
      std_msgs::msg::Float32MultiArray tool_grasping_position_msg;
      tool_grasping_position_msg.data = {toolgrasp_centroid(0),toolgrasp_centroid(1),toolgrasp_centroid(2),roll_command_deg,pitch_command_deg};
      tool_grasping_pub->publish(tool_grasping_position_msg);
      handover_trigged_msg.data = false;

    } else {
      RCLCPP_INFO(get_logger(), "Tool not in hand");
    }
    


    // convert pcl to Pointcloud2
    //processed_cloud->width = 1;
    //processed_cloud->height = processed_cloud->points.size();
    sensor_msgs::msg::PointCloud2 processed_cloud_msg;
    pcl::toROSMsg(*processed_cloud, processed_cloud_msg);
    processed_cloud_msg.header = msg->header;

    // Publish the processed point cloud
    pcl_publisher->publish(processed_cloud_msg);
  }

  void processFloatArray(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
  {
    // Process float array data here
    
    right_hand_x = msg->data[0];
    right_hand_y = msg->data[1];
    right_hand_z = msg->data[2];
    right_elbow_x = msg->data[3];
    right_elbow_y = msg->data[4];
    right_elbow_z = msg->data[5];
    //printf("%f,%f,%f,%f,%f,%f \n",right_hand_x,right_hand_y,right_hand_z,right_elbow_x,right_elbow_y,right_elbow_z);
    //RCLCPP_INFO(get_logger(), "wrist:%.4f,%.4f,%.4f,elbow:%.4f,%.4f,%.4f", right_hand_x,right_hand_y,right_hand_z,right_elbow_x,right_elbow_y,right_elbow_z);

    // check right hand is in workcell
    //printf("xyz: %.4f, %.4f, %.4f \n", right_hand_x, right_hand_y, right_hand_z);
    if (is_in_the_cell()){
      float distance = std::sqrt(std::pow(right_hand_x - prev_x, 2) +
                                 std::pow(right_hand_y - prev_y, 2) +
                                 std::pow(right_hand_z - prev_z, 2));
      

      if (distance > 0.01){
        timer_duration = 0;
      } else {
        timer_duration += (this->now() - prev_time).seconds();
      }

      if (timer_duration > static_hand_duration){
        //RCLCPP_INFO(this->get_logger(), "Handover triggered!!!!!!!!!!!!!!!!!!");
        
        handover_trigged_msg.data = true;
        handover_bool_pub->publish(handover_trigged_msg);
      }

      // Update previous position and time
      prev_x = right_hand_x;
      prev_y = right_hand_y;
      prev_z = right_hand_z;
      prev_time = this->now();
      handover_trigged_msg.data = false;
    }
  }


};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SubscriberNode>());
  rclcpp::shutdown();
  return 0;
}

