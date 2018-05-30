#include <ros/ros.h>
#include <sensor_msgs/Range.h>
#include <std_msgs/Int16.h>
#include <fstream>
#include <ctime>
#include <time.h>
#include <sstream>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Float64.h>
#include <nav_msgs/Odometry.h>
#include <tf/tf.h>
#include <math.h>

double current_t;
double current_t_imu;
double last_t_imu;
double roll, pitch, yaw;

double const SET_VAL = 0.5; // set distance to wall in m

const double PK = 800.0; //  proportional gain
const double PD = 80.0; // differential gain
double err, last_err;
double r_range, l_range, f_range; //us sensor values
double last_t;
int m_out, s_out;

bool inCorner, yaw_initialized; // corner status
double turn, yaw_at_corner_entry;

void initializeValues(){ //initialisation
    err = 0.0;
    last_err = 0.0;
    r_range = 0.0;
    l_range = 0.0;
    f_range = 0.0;
    last_t = ((double)clock()/CLOCKS_PER_SEC);
    inCorner = false;
    yaw_initialized = false;
    turn = 0.0;
    yaw_at_corner_entry = 0.0;
}

bool crashed(){
    return (r_range < 0.1 && r_range != 0) || (l_range < 0.1 && l_range != 0) || (f_range < 0.2 && f_range != 0);
}

void limit_output(){
    if (s_out < -800)
        s_out = -800;
    if (s_out > 800)
        s_out = 800;
}

// callback functions get called whenever a new message is availible in the input puffer
void uslCallback(sensor_msgs::Range::ConstPtr uslMsg, sensor_msgs::Range* usl)
{
    *usl = *uslMsg;
}

void usfCallback(sensor_msgs::Range::ConstPtr usfMsg, sensor_msgs::Range* usf)
{
    *usf = *usfMsg;
}

void usrCallback(sensor_msgs::Range::ConstPtr usrMsg, sensor_msgs::Range* usr)
{
    *usr = *usrMsg;
    current_t = ((double)clock()/CLOCKS_PER_SEC);
}

void imuCallback(sensor_msgs::Imu::ConstPtr imuMsg, sensor_msgs::Imu* imu)
{
    *imu = *imuMsg;
    current_t_imu = ((double)clock()/CLOCKS_PER_SEC);
}

void hallDtCallback(std_msgs::Float64::ConstPtr msg, std_msgs::Float64* out) {
    *out = *msg;
}

void odomCallback(nav_msgs::Odometry::ConstPtr msg, nav_msgs::Odometry* out) {
    *out = *msg;
    // convert quaternions to euler angles
    tf::Quaternion q;
    tf::quaternionMsgToTF(msg->pose.pose.orientation, q);
    tf::Matrix3x3 mat(q);
    mat.getEulerYPR(yaw, pitch, roll);
}

int main(int argc, char** argv)
{
    // init this node
    ros::init(argc, argv, "helloworld_node");
    // get ros node handle
    ros::NodeHandle nh;

    // sensor message container
    sensor_msgs::Range usr, usf, usl;
    sensor_msgs::Imu imu;
    std_msgs::Int16 motor, steering;
    std_msgs::Float64 hallDt;
    nav_msgs::Odometry odom;

    // generate subscriber for sensor messages
    ros::Subscriber usrSub = nh.subscribe<sensor_msgs::Range>(
                "/uc_bridge/usr", 10, boost::bind(usrCallback, _1, &usr));
    ros::Subscriber uslSub = nh.subscribe<sensor_msgs::Range>(
                "/uc_bridge/usl", 10, boost::bind(uslCallback, _1, &usl));
    ros::Subscriber usfSub = nh.subscribe<sensor_msgs::Range>(
                "/uc_bridge/usf", 10, boost::bind(usfCallback, _1, &usf));
    ros::Subscriber imuSub = nh.subscribe<sensor_msgs::Imu>(
                "/uc_bridge/imu", 10, boost::bind(imuCallback, _1, &imu));
    ros::Subscriber hallDtSub = nh.subscribe<std_msgs::Float64>(
                "/uc_bridge/hallDt", 10, boost::bind(hallDtCallback, _1, &hallDt));
    ros::Subscriber odomSub = nh.subscribe<nav_msgs::Odometry>(
                "/odom", 10, boost::bind(odomCallback, _1, &odom));

    // generate control message publisher
    ros::Publisher motorCtrl =
            nh.advertise<std_msgs::Int16>("/uc_bridge/set_motor_level_msg", 1);
    ros::Publisher steeringCtrl =
            nh.advertise<std_msgs::Int16>("/uc_bridge/set_steering_level_msg", 1);

    initializeValues();

    ros::Rate loop_rate(50);
    while (ros::ok())
    {
        // Check for new sensor data
        if (usr.range != r_range) {
            // Grab current values for further processing
            r_range = usr.range;
            l_range = usl.range;
            f_range = usf.range;

            // Avoid crashing on each side
            if(!crashed()){
                if (r_range == 0) {
                    r_range = 5.0;
                }

                //Corner detection
                if (inCorner) {
                    if(!yaw_initialized){
                        yaw_at_corner_entry = yaw;
                        yaw_initialized = true;
                    }

                    turn = yaw;

                    // Overflow
                    if(yaw_at_corner_entry < -3.14+1.4){
                        int difference_to_overflow = fabs(-3.14-yaw_at_corner_entry); //calculate positive difference
                        int rest = 1.4-difference_to_overflow; //rest, which still has to be steered after overflow
                        int new_threshold = 3.14-rest; //If new_threshold is smaller then end of corner detected

                        if(turn < new_threshold && turn>yaw_at_corner_entry){
                            ROS_INFO("CORNER END - SWITCH TO PD CONTROLLER");
                            inCorner = false;
                        }

                        else {
                            ROS_INFO("OVERFLOW!!! CORNER CONTROLLER: yaw_at_corner=%f, yaw=%f",yaw_at_corner_entry,turn);
                            s_out = 800;
                            m_out = 200;
                        }
                    }
                    else {
                        if (turn < yaw_at_corner_entry-1.4) { //End of corner detected, driven 90 degrees (PI/2)
                            ROS_INFO("CORNER END - SWITCH TO PD CONTROLLER");
                            inCorner = false;
                        }
                        else {
                            ROS_INFO("CORNER CONTROLLER: yaw_at_corner=%f, yaw=%f",yaw_at_corner_entry,turn);
                            s_out = 800;
                            m_out = 200;
                        }
                    }
                }

                //Wall PD-Controller
                else {
                    if (r_range > 1.5) { //Corner detected
                        ROS_INFO("WALL END - SWITCH TO CORNER CONTROLLER");
                        inCorner = true;
                        yaw_initialized = false;
                        s_out = 800;
                        m_out = 200;
                    }

                    else { // no corner detected
                        ROS_INFO("PD CONTROLLER");
                        last_err = err;
                        err = SET_VAL - r_range;

                        // constant motor speed
                        m_out = 400; // Risky variant: 450
                        // PD-Controller for steering angle (right)
                        s_out = -(int)(PK*err + PD * 300 / m_out * (err - last_err)/(current_t - last_t));

                        last_t = current_t;
                    }
                }
            }
            else{ //avoid crash
                s_out = 0;
                m_out = 0;
            }

            // limit steering output
            limit_output();
        }

        motor.data = m_out;
        steering.data = s_out;

        // publish command messages on their topics
        motorCtrl.publish(motor);
        steeringCtrl.publish(steering);

        // clear input/output buffers
        ros::spinOnce();
        // this is needed to ensure a const. loop rate
        loop_rate.sleep();
    }

    ros::spin();
}
