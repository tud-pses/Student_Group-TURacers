#include <ros/ros.h>
#include <sensor_msgs/Range.h>
#include <std_msgs/Int16.h>
#include <fstream>
#include <ctime>
#include <time.h>
#include <sstream>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Float64.h>
#include <nav_msgs/Odometry.h>
#include <tf/tf.h>
#include <math.h>
#include <cv_bridge/cv_bridge.h>


// Variables
double current_t;
double current_t_imu;
double roll, pitch, yaw;
double setValRight; // actual setValue for wall-follow-controller
double setValLeft; // actual setValue for wall-follow-controller
int m_out, s_out;

double err, last_err;
double r_range, l_range, f_range; // values of range sensors
double last_r_range, last_l_range;
double last_t;

bool yaw_initialized;
int inCorner;
bool RecommendedDistanceReceived;

int controllerState;
int avoidingObstacle;
double oldRecommendedDistance;
double turn, yaw_at_corner_entry;

clock_t begin;

// Constants
const double MIN_WAIT_TIME = 0.25;// was in min = 15 sec

const double PK = 800.0; // proportional gain
const double PD = 80.0; // differential gain

const int FAST = 250;
const int SLOW = 200;

const double DEFAULT_SET_VAL = 0.5;

// States for avoidingOpstacle
const int NO_OBJECT = 0;
const int DETECTING = 1;
const int PASS_RIGHT = 2;
const int PASS_LEFT = 3;
const int PASSING_RIGHT = 4;
const int PASSING_LEFT = 5;
const int WALL = 6;
const int SLOWRETURN = 7;

// States for controllerState
const int RIGHT = 0;
const int LEFT = 1;
const int TUNNEL = 2;

// States for inCorner
const int NO_CORNER = 0;
const int DETECTING_CORNER = 1;
const int IN_CORNER = 2;

void initializeValues(){
    setValRight = DEFAULT_SET_VAL;
    setValLeft = DEFAULT_SET_VAL;

    err = 0.0;
    last_err = 0.0;
    r_range = 0.0;
    l_range = 0.0;
    f_range = 0.0;
    last_t = ((double)clock()/CLOCKS_PER_SEC);
    inCorner = NO_CORNER;
    yaw_initialized = false;
    turn = 0.0;
    yaw_at_corner_entry = 0.0;

    avoidingObstacle = NO_OBJECT;
    RecommendedDistanceReceived = false;
    m_out = FAST;
    controllerState = RIGHT;
}

void setsetValRight(double newsetValRight) {
    setValRight = newsetValRight;
    ROS_INFO("new setValRight = %f",setValRight);
}

void setsetValLeft(double newsetValLeft) {
    setValLeft = newsetValLeft;
    ROS_INFO("new setValLeft = %f",setValLeft);
}

// Callbacks get called whenever a new message is availible in the input puffer
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

void odomCallback(nav_msgs::Odometry::ConstPtr msg, nav_msgs::Odometry* out) {
    *out = *msg;
    //convert quaternions to euler angles
    tf::Quaternion q;
    tf::quaternionMsgToTF(msg->pose.pose.orientation, q);
    tf::Matrix3x3 mat(q);
    mat.getEulerYPR(yaw, pitch, roll);
}

void obstacleCallback(std_msgs::Float64::ConstPtr msg, std_msgs::Float64* out) {
    *out = *msg;
    RecommendedDistanceReceived = true;
}

void obstacleStateCallback(std_msgs::Int16::ConstPtr msg, std_msgs::Int16 * state)
{
    *state = *msg;
}

int main(int argc, char** argv)
{
    // init this node
    ros::init(argc, argv, "wall_follow_obstacle_node");
    // get ros node handle
    ros::NodeHandle nh;

    // sensor message container
    sensor_msgs::Range usr, usf, usl;
    sensor_msgs::Imu imu;
    std_msgs::Int16 motor, steering, obstacleState;
    std_msgs::Float64 recommendedDistance;
    nav_msgs::Odometry odom;
    sensor_msgs::Image::ConstPtr depth_img_ros;
    cv::Mat im16;

    // generate subscriber for sensor messages
    ros::Subscriber usrSub = nh.subscribe<sensor_msgs::Range>(
                "/uc_bridge/usr", 10, boost::bind(usrCallback, _1, &usr));
    ros::Subscriber uslSub = nh.subscribe<sensor_msgs::Range>(
                "/uc_bridge/usl", 10, boost::bind(uslCallback, _1, &usl));
    ros::Subscriber usfSub = nh.subscribe<sensor_msgs::Range>(
                "/uc_bridge/usf", 10, boost::bind(usfCallback, _1, &usf));
    ros::Subscriber imuSub = nh.subscribe<sensor_msgs::Imu>(
                "/uc_bridge/imu", 10, boost::bind(imuCallback, _1, &imu));
    ros::Subscriber odomSub = nh.subscribe<nav_msgs::Odometry>(
                "/odom", 10, boost::bind(odomCallback, _1, &odom));
    ros::Subscriber obstacleSub = nh.subscribe<std_msgs::Float64>("wall_follow_obstacle/recommended_distance_msg", 1, boost::bind(obstacleCallback, _1, &recommendedDistance));

    ros::Subscriber obstacleStateSub = nh.subscribe<std_msgs::Int16>("wall_follow_obstacle/avoidung_state_msg", 1, boost::bind(obstacleStateCallback, _1, &obstacleState));


    // generate control message publisher
    ros::Publisher motorCtrl =
            nh.advertise<std_msgs::Int16>("/uc_bridge/set_motor_level_msg", 1);
    ros::Publisher steeringCtrl =
            nh.advertise<std_msgs::Int16>("/uc_bridge/set_steering_level_msg", 1);

    initializeValues();

    // Loop starts here:
    // loop rate value is set in Hz
    ros::Duration(5).sleep(); //wait for detect obstacle initialisation
    ros::Rate loop_rate(50);
    while (ros::ok())
    {

        if (usr.range != r_range) {

            // avoid crashing
            if (usf.range < 0.25 && usf.range != 0)
            {
                m_out = 0;
            }
            else {
                m_out = SLOW;
                if (usr.range == 0) {
                    r_range = 5.0;
                }
                else {
                    r_range = usr.range;
                }
                if (usl.range == 0) {
                    l_range = 5.0;
                }
                else {
                    l_range = usl.range;
                }
                if (usf.range == 0) {
                    f_range = 5.0;
                }
                else {
                    f_range = usf.range;
                }

                //Corner detection
                if (inCorner == IN_CORNER) {
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
                            inCorner = NO_CORNER;
                            avoidingObstacle = NO_OBJECT;
                        }

                        else {
                            s_out = 800;
                            m_out = SLOW;
                        }
                    }
                    else {
                        if (turn < yaw_at_corner_entry-1.4) {
                            inCorner = NO_CORNER;
                            avoidingObstacle = NO_OBJECT;
                        }
                        else {
                            s_out = 800;
                            m_out = SLOW;
                        }
                    }
                }

                //Wall PD-Controller
                else {
                    if (inCorner == NO_CORNER && r_range > 1.7 && avoidingObstacle != PASSING_LEFT && s_out > -200 && r_range != 5.0) { //Corner detected and not avoiding obstacle on the left side.
                        inCorner = DETECTING_CORNER;
                        ROS_INFO("Detecting Corner");
                    }
                    else if(inCorner == DETECTING_CORNER) {
                        if(r_range > 1.7 && avoidingObstacle != PASSING_LEFT && s_out > -200 && r_range != 5.0){
                            inCorner = IN_CORNER;
                            yaw_initialized = false;

                            setsetValRight(DEFAULT_SET_VAL);//return to normal Course if corner is detected (after cornering is completed)
                            controllerState = RIGHT;
                            avoidingObstacle = NO_OBJECT;

                            s_out = 800;
                            m_out = SLOW;
                            ROS_INFO("in Corner");
                        }
                        else {
                            inCorner = NO_CORNER;
                            ROS_INFO("no Corner");

                        }

                    }
                    else {//no corner detected

                        //Obstacle Detection

                        double avoiding_wall_distance;

                        avoiding_wall_distance = recommendedDistance.data;

                        switch (avoidingObstacle){ //state machine: switches states according to postion, obstacles and planned path
                        case SLOWRETURN: //return to intermediate value before returning to default setting
                            controllerState = RIGHT;
                            m_out = FAST;
                            if(r_range < 0.9) {
                                setsetValRight(DEFAULT_SET_VAL);
                                avoidingObstacle = NO_OBJECT;
                            }
                        case NO_OBJECT: //no object detected, default. Gets executed in NO_OBJECT and SLOWRETURN state
                            controllerState = RIGHT;
                            m_out = FAST;
                            if(RecommendedDistanceReceived && obstacleState.data != NO_OBJECT && obstacleState.data != WALL && std::abs(r_range - last_r_range) < 0.01) {
                                //if obstacle detected and not actually avoiding and driving straight
                                avoidingObstacle = DETECTING;
                                oldRecommendedDistance = avoiding_wall_distance;
                                ROS_INFO("avoidingObstacle state = DETECTING");
                            }
                            break;
                        case DETECTING: //obstacle detected once, confirming posititon
                            controllerState = RIGHT;
                            m_out = FAST;
                            if(RecommendedDistanceReceived) {

                                if(avoiding_wall_distance <= oldRecommendedDistance + 0.1 && avoiding_wall_distance >= oldRecommendedDistance - 0.1 && std::abs(r_range - last_r_range) < 0.01){

                                    if(obstacleState.data == PASS_RIGHT) {//passing obstacle on right
                                        setsetValRight(avoiding_wall_distance);
                                        avoidingObstacle = PASS_RIGHT;
                                        ROS_INFO("avoidingObstacle state = PASS_RIGHT");
                                        begin = clock();
                                    }
                                    else if (obstacleState.data == PASS_LEFT && r_range != 5.0 && l_range != 5.0 && l_range - (avoiding_wall_distance - r_range) > 0.2){//passing obstacle on left
                                        setsetValRight(avoiding_wall_distance);
                                        avoiding_wall_distance = l_range - (avoiding_wall_distance - r_range);
                                        controllerState = LEFT;
                                        setsetValLeft(avoiding_wall_distance);
                                        avoidingObstacle = PASS_LEFT;
                                        ROS_INFO("avoidingObstacle state = PASS_LEFT");
                                        begin = clock();
                                    }
                                }
                                else {//Not similar value
                                    avoidingObstacle = NO_OBJECT;
                                    ROS_INFO("avoidingObstacle state = NO_OBJECT no similar value. avoiding_wall_distance = %f, oldRecommendedDistance = %f", avoiding_wall_distance, oldRecommendedDistance);
                                }
                            }
                            break;
                        case PASS_LEFT: //passing obstacle  on left
                            controllerState = LEFT;
                            m_out = SLOW;

                            if(last_r_range - r_range > 0.2 && r_range < 0.6) {
                                avoidingObstacle = PASSING_LEFT; // state: next to object
                                ROS_INFO("avoidingObstacle state = PASSING_LEFT");
                            }
                            else if(((double)clock() - (double)begin) / (double)CLOCKS_PER_SEC > MIN_WAIT_TIME && r_range > 0.2) {
                                setsetValRight(DEFAULT_SET_VAL);//return to normal Course
                                avoidingObstacle = NO_OBJECT; // state: normal mode
                                ROS_INFO("avoidingObstacle state = NO_OBJECT");
                                if(setValRight > 1.0){
                                    setsetValRight(0.8);
                                    avoiding_wall_distance = SLOWRETURN;
                                    ROS_INFO("avoidingObstacle state = SLOWRETURN");
                                }
                            }

                            break;
                        case PASS_RIGHT: //passing obstacle  on right
                            controllerState = RIGHT;
                            m_out = SLOW;
                            //              ROS_INFO("last_r_range - r_range = %f time = %f", last_r_range - r_range, ((double)clock() - (double)begin) / (double)CLOCKS_PER_SEC);
                            if(avoiding_wall_distance < setValRight && std::abs(r_range - last_r_range) < 0.01 && obstacleState.data == PASS_RIGHT){
                                setsetValRight(avoiding_wall_distance);
                            }
                            if(last_l_range - l_range < 0.2 && l_range < 0.6) {
                                avoidingObstacle = PASSING_RIGHT; // state: next to object
                                ROS_INFO("avoidingObstacle state = PASSING_RIGHT");
                            }
                            else if(((double)clock() - (double)begin) / (double)CLOCKS_PER_SEC > MIN_WAIT_TIME) {
                                setsetValRight(DEFAULT_SET_VAL);//return to normal Course
                                avoidingObstacle = NO_OBJECT; // state: normal mode
                                ROS_INFO("avoidingObstacle state = NO_OBJECT");
                            }
                            break;
                        case PASSING_LEFT: //left of obstacle
                            controllerState = LEFT;
                            //              ROS_INFO("last_r_range - r_range = %f", last_r_range - r_range);
                            m_out = SLOW;

                            if (last_r_range - r_range < 0.2 || ((double)clock() - (double)begin) / (double)CLOCKS_PER_SEC > MIN_WAIT_TIME) {
                                if(setValRight > 1.0){
                                    setsetValRight(0.8);
                                    avoiding_wall_distance = SLOWRETURN;
                                    ROS_INFO("avoidingObstacle state = SLOWRETURN");

                                }else {
                                    setsetValRight(DEFAULT_SET_VAL);//return to normal Course
                                    avoidingObstacle = NO_OBJECT; // state: normal mode
                                    ROS_INFO("avoidingObstacle state = NO_OBJECT");
                                }
                            }
                            break;
                        case PASSING_RIGHT: //right of obstacle
                            controllerState = RIGHT;
                            //              ROS_INFO("last_l_range - l_range = %f", last_l_range - l_range);
                            m_out = SLOW;

                            if (last_l_range - l_range > 0.2 || ((double)clock() - (double)begin) / (double)CLOCKS_PER_SEC > MIN_WAIT_TIME) {
                                setsetValRight(DEFAULT_SET_VAL);//return to normal Course
                                avoidingObstacle = NO_OBJECT; // state: normal mode
                                ROS_INFO("avoidingObstacle state = NO_OBJECT");
                            }
                            break;
                        }
                        RecommendedDistanceReceived = false;
                        last_r_range = r_range;
                        last_l_range = l_range;

                        last_err = err;

                        // constant motor speed
                        // PD-Controller for steering angle

                        switch(controllerState) { //state machine: switches controller between a tunnel scenario, left and right wall
                        case RIGHT: //controller uses distance to right wall

                            if(l_range < 0.2 && r_range < 0.2) {
                                ROS_INFO("Tunnel detected");
                                controllerState = TUNNEL;
                                setValRight = (r_range + l_range) / 2;
                            }

                            err = setValRight - r_range;
                            s_out = -(int)(PK*err + PD * (err - last_err) / (current_t - last_t)); //300 / m_out
                            break;
                        case LEFT: //controller uses distance to left wall

                            if(l_range < 0.2 && r_range < 0.2) {
                                ROS_INFO("Tunnel detected");
                                controllerState = TUNNEL;
                                setValRight = (r_range + l_range) / 2;
                            }

                            err = setValLeft - l_range;
                            s_out = (int)(PK*err + PD * (err - last_err) / (current_t - last_t)); //300 / m_out
                            break;
                        case TUNNEL: //controller uses distance to left and right wall
                            if(l_range > 0.4) {
                                controllerState = RIGHT;
                                setsetValRight(DEFAULT_SET_VAL);
                                avoidingObstacle = NO_OBJECT;
                            }

                            err = setValRight - r_range;
                            s_out = -(int)(PK*err + PD * (err - last_err) / (current_t - last_t)); //300 / m_out
                            break;
                        }

                        // limit steering output for controller (not for cornering mode)
                        if (s_out < -600) {
                            s_out = -600;
                        }
                        if (s_out > 600) {
                            s_out = 600;
                        }

                        last_t = current_t;
                    }
                }
            }



            // limit steering output (mechanical limits, security feature, not actually needed)
            if (s_out < -800) {
                s_out = -800;
            }
            if (s_out > 800) {
                s_out = 800;
            }
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
