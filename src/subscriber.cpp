#include "ros/ros.h"
#include "std_msgs/String.h"
#include "geometry_msgs/Twist.h"
#include <urdf/model.h>
#include <iostream>

using namespace std;

void handleTurn(const geometry_msgs::Twist::ConstPtr& msg) {
	//ROS_INFO("I heard: [%s]", msg->data.c_str());
	cout << "twist linear: " << msg->linear.x << endl;
	cout << "twist angular: " << msg->angular.z << endl;
	if (msg->angular.z < 0) {
		cout << "turning right" << endl;
	}
	else if (msg->angular.z > 0) {
		cout << "turning left" << endl;
	}
	else {
		cout << "going straight ahead" << endl;
	}
}

int main(int argc, char **argv) {
	ros::init(argc, argv, "subscriber");
	std::string urdf_file = "./src/wheelchair_description/urdf/wheelchair.urdf";

	urdf::Model model;
	if (!model.initFile(urdf_file)) {
		ROS_ERROR("Failed to parse urdf file");
		return -1;
	}
	ROS_INFO("Successfully parsed urdf file");

	ros::NodeHandle n;

	// through subscribe() you tell ROs that you want to receive messages from the given topic
	// messages are passed to a callback function
	ros::Subscriber sub = n.subscribe("/cmd_vel", 1000, handleTurn);

	ros::spin();
	return 0;
}