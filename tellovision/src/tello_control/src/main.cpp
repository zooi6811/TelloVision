#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/empty.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/executors/multi_threaded_executor.hpp"

// for multi callbacks
#include "rclcpp/callback_group.hpp"
#include "rclcpp/subscription_options.hpp"


#define PI 3.14159265359
#define PI2 PI * 2.0
#define DEG_TO_RAD PI / 180

#define NO_KEY -1

#define KEY_NUM_0 48
#define KEY_NUM_1 49
#define KEY_NUM_2 50
#define KEY_NUM_3 51
#define KEY_NUM_4 52
#define KEY_NUM_5 53
#define KEY_NUM_6 54
#define KEY_NUM_7 55
#define KEY_NUM_8 56
#define KEY_NUM_9 57

#define KEY_UP 82
#define KEY_DOWN 84
#define KEY_LEFT 81
#define KEY_RIGHT 83

#define KEY_ENTER 13
#define KEY_SPACE 32

#define KEY_M 109

using namespace std::chrono_literals;

class TelloControl : public rclcpp::Node
{
	public:
		// Storage for the last key pressed
		int last_key = NO_KEY;

		// Timer for the main loop
		rclcpp::TimerBase::SharedPtr timer;

		// Publisher for drone velocity commands
		rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_velocity;

		// Publisher for drone takeoff commands
		rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr publisher_takeoff;

		// Publisher for drone flip commands
		rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_flip;

		// Publisher for drone landing commands
		rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr publisher_land;

		// Publisher for emergency stop
		rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr publisher_emergency;

		// extra command
		rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_move;
		
		// // Subscription for hand gestures
		rclcpp::Subscription<std_msgs::msg::String>::SharedPtr gesture_subscriber_;

		/**
		 * Construct a new Tello Control object
		 */
		TelloControl() : Node("control")
		{
			publisher_land = this->create_publisher<std_msgs::msg::Empty>("land", 1);
			publisher_flip = this->create_publisher<std_msgs::msg::String>("flip", 1);
			publisher_takeoff = this->create_publisher<std_msgs::msg::Empty>("takeoff", 1);
			publisher_velocity = this->create_publisher<geometry_msgs::msg::Twist>("control", 1);
			publisher_emergency = this->create_publisher<std_msgs::msg::Empty>("emergency", 1);

			// Publisher for move commands
			publisher_move = this->create_publisher<std_msgs::msg::String>("move", 10);


			// OpenCV window once
			cv::namedWindow("Tello", cv::WINDOW_AUTOSIZE);

		
			gesture_subscriber_ = this->create_subscription<std_msgs::msg::String>(
				"/bob",
				10,
				std::bind(&TelloControl::handGestureCallback, this, std::placeholders::_1)
				
			);

		}

		// callback for gesture control
		void handGestureCallback(const std_msgs::msg::String::SharedPtr msg) const
		{
			RCLCPP_INFO(this->get_logger(), "Received gesture: '%s'", msg->data.c_str());
			if (msg->data == "move_up") {
				std_msgs::msg::String m;
				m.data = "up 30";
				publisher_move->publish(m);
				RCLCPP_INFO(this->get_logger(), "Move command: %s", m.data.c_str());
			}
			if (msg->data == "move_down") {
				std_msgs::msg::String m;
				m.data = "down 30";
				publisher_move->publish(m);
				RCLCPP_INFO(this->get_logger(), "Move command: %s", m.data.c_str());
			}
			if (msg->data == "move_left") {
				std_msgs::msg::String m;
				m.data = "right 50";
				publisher_move->publish(m);
				RCLCPP_INFO(this->get_logger(), "Move command: %s", m.data.c_str());
			}
			if (msg->data == "move_right") {
				std_msgs::msg::String m;
				m.data = "left 50";
				publisher_move->publish(m);
				RCLCPP_INFO(this->get_logger(), "Move command: %s", m.data.c_str());
			}
			if (msg->data == "land") {
				RCLCPP_INFO(get_logger(), "Landing emergency stop via keypress");
				publisher_land->publish(std_msgs::msg::Empty());
			}
			if (msg->data == "move_back") {
				std_msgs::msg::String m;
				m.data = "back 50";
				publisher_move->publish(m);
				RCLCPP_INFO(this->get_logger(), "Move command: %s", m.data.c_str());
			}
			if (msg->data == "move_forward") {
				std_msgs::msg::String m;
				m.data = "forward 30";
				publisher_move->publish(m);
				RCLCPP_INFO(this->get_logger(), "Move command: %s", m.data.c_str());
			}
			if (msg->data == "flip_back") {
				std_msgs::msg::String m;
				m.data = "b";
				publisher_flip->publish(m);
				RCLCPP_INFO(this->get_logger(), "Flip command: %s", m.data.c_str());
			}
		}

		/**
		 * Method to control the drone using the keyboard inputs.
		 *
		 * @param key Keycode received.
		 */
		void manualControl(int key)
		{
			// Speed of the drone in manual control mode.
			double manual_speed = 50;

			geometry_msgs::msg::Twist msg = geometry_msgs::msg::Twist();
		
			if(key == KEY_LEFT) {msg.linear.x = -manual_speed;}
			if(key == KEY_RIGHT) {msg.linear.x = +manual_speed;}
			if(key == KEY_UP) {msg.linear.y = manual_speed;}
			if(key == KEY_DOWN) {msg.linear.y = -manual_speed;}
			if(key == (int)('w')) {msg.linear.z = manual_speed;}
			if(key == (int)('s')) {msg.linear.z = -manual_speed;}
			if(key == (int)('a')) {msg.angular.z = -manual_speed;}
			if(key == (int)('d')) {msg.angular.z = manual_speed;}
			// adding extra key to test
			if(key == (int)('t')){
				RCLCPP_INFO(get_logger(), "taking off");
				publisher_takeoff->publish(std_msgs::msg::Empty());
				std_msgs::msg::String m;
				m.data = "up 80";
				publisher_move->publish(m);
			}
			if(key == (int)('l')){
				// msg.angular.z = 30;
				// msg.linear.x = -30;
				RCLCPP_INFO(get_logger(), "Landing emergency stop via keypress");
				publisher_land->publish(std_msgs::msg::Empty());
			}
			if (key == (int)('z')) {
				std_msgs::msg::String m;
				m.data = "forward 100";
				publisher_move->publish(m);
				RCLCPP_INFO(this->get_logger(), "Move command: %s", m.data.c_str());
				return;  // skip the velocity publish
			}
			if (key == (int)('x')) {
				std_msgs::msg::String m;
				m.data = "back 50";
				publisher_move->publish(m);
				RCLCPP_INFO(this->get_logger(), "Move command: %s", m.data.c_str());
				return;  // skip the velocity publish
			
			}
			if (key == (int)('m')) {
				std_msgs::msg::String m;
				m.data = "right 50";
				publisher_move->publish(m);
				RCLCPP_INFO(this->get_logger(), "Move command: %s", m.data.c_str());
				return;  // skip the velocity publish
			}
			if (key == (int)('n')) {
				std_msgs::msg::String m;
				m.data = "left 50";
				publisher_move->publish(m);
				RCLCPP_INFO(this->get_logger(), "Move command: %s", m.data.c_str());
				return;  // skip the velocity publish
			}

		
			publisher_velocity->publish(msg);
		}



};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TelloControl>();

  // 1) Spin ROS in the background
  std::thread ros_thread([&](){
    rclcpp::executors::MultiThreadedExecutor exec;
    exec.add_node(node);
    exec.spin();
  });

  // 2) Create and drive the GUI in main thread
  cv::namedWindow("Tello", cv::WINDOW_AUTOSIZE);
  while (rclcpp::ok()) {
    // render your frame
    cv::Mat frame = cv::Mat::zeros(200, 200, CV_8UC3);
    cv::imshow("Tello", frame);

    // poll key and call manualControl directly
    int key = cv::waitKey(33);  // ~30Hz
    if (key != -1) {
      node->manualControl(key);
    }
  }

  ros_thread.join();
  rclcpp::shutdown();
  return 0;
}
