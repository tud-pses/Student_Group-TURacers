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
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <image_transport/image_transport.h>


//opencv
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"


bool imageReceived;
cv::Mat refImage, maskImageThresh, imCam, lastIm, newMaskImage; // Variable to save images
sensor_msgs::ImagePtr binaryImageVar, rectImageVar;
bool saveImages, publishImages, adjustImages;

int rightRange, leftRange; // in cm
sensor_msgs::Range usr, usl;

int avoidingState;
cv::Rect boundingBox;
cv::Rect rectFreeSpaceRight;
cv::Rect rectFreeSpaceLeft;

const double TAN_35 = std::tan(35);

const int HALF_CAR_WIDTH = 13; // in cm
const int MIN_OBJ_DISTANCE_TO_WALL = 50; // in cm;
const double DEFAULT_SET_VAL = 0.5; // in m

const int NO_OBJECT = 0;
const int PASS_RIGHT = 2;
const int PASS_LEFT = 3;
const int WALL = 6;

void initializeValues(){
    imageReceived = false;
    cv::Mat refImgSrc, refImage_im16, maskImgSrc, maskImage_im8, newMaskImageSrc;


    refImgSrc = cv::imread("/home/pses/catkin_ws/src/pses-turacers/detect_obstacle/src/refImage3_PNG.png", cv::IMREAD_UNCHANGED);
    maskImgSrc = cv::imread("/home/pses/catkin_ws/src/pses-turacers/detect_obstacle/src/maskImage3_PNG.png", cv::IMREAD_UNCHANGED);
    newMaskImageSrc = cv::imread("/home/pses/catkin_ws/src/pses-turacers/detect_obstacle/src/newMaskImage.png", cv::IMREAD_UNCHANGED);

    refImgSrc.convertTo(refImage_im16, CV_16UC1);
    maskImgSrc.convertTo(maskImage_im8, CV_8UC1);
    newMaskImageSrc.convertTo(newMaskImage, CV_8UC1);
    cv::threshold(maskImage_im8, maskImageThresh, 50, 255,cv::THRESH_BINARY);
    refImage_im16.copyTo(refImage, maskImageThresh);

    saveImages = false;
    publishImages = false;
    adjustImages = false;
    lastIm = refImage;
    rightRange = 0;
    leftRange = 0;
    avoidingState = NO_OBJECT;

}

double detectObstacle(cv::Mat im16){

    cv::Mat im8;
    im16.convertTo(im8, refImage.depth()); //convert input image to 8-bit

    cv::Mat diffImage, filteredImg;

    cv::subtract(lastIm, im16, diffImage); //subtract image from previous one

    lastIm = im16;
    cv::Mat diffImage_im8;
    diffImage.convertTo(diffImage_im8, CV_8UC1); //convert difference image to 8-bit

    // calculate binary image
    cv::Mat binImage, binImage_masked, threshedImage;


    cv::threshold( diffImage_im8, threshedImage, 25, 255,cv::THRESH_TOZERO_INV ); //delete noise
    cv::threshold( threshedImage, binImage, 6, 255,cv::THRESH_BINARY ); //threshold to get obstacles
    binImage.copyTo(binImage_masked,newMaskImage);

    cv::erode(binImage_masked, filteredImg, cv::Mat()); //delete noise particles

    std::vector<std::vector<cv::Point> > contours0;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(filteredImg, contours0, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    // draw contours --> box
    cv::Mat drawing;
    if(saveImages || publishImages) {
        drawing = cv::Mat::zeros(filteredImg.rows, filteredImg.cols, CV_8UC3);
    }

    unsigned long lastDistanceToObject = 65535;
    unsigned long distanceToObject;
    unsigned long distanceFreeSpaceRight;
    unsigned long distanceFreeSpaceLeft;

    double returnDistance = DEFAULT_SET_VAL; //returns default if no obstacle detected
    avoidingState = NO_OBJECT;

    for( int i = 0; i< contours0.size(); i++ ) //iterates over all possible obstacles
    {
        float ctArea= cv::contourArea(contours0[i]);
        if(ctArea > 1200)
        { //obstacle detected

            // compute the rotated bounding rect of the biggest contour
            boundingBox = cv::boundingRect(contours0[i]);

            distanceToObject = 0;

            int nonzeroPx = 0;
            cv::Mat obstacleImg = im16(boundingBox);
            for (int r = 0; r < obstacleImg.rows; ++r) { // calculates average of distance to object
                for (int c = 0; c < obstacleImg.cols; ++c) {
                    if (obstacleImg.at<unsigned short>(r,c) != 0){
                        distanceToObject += obstacleImg.at<unsigned short>(r,c);
                        nonzeroPx++;
                    }
                }
            }
            if(nonzeroPx != 0) { //avoids division by zero
                distanceToObject = distanceToObject / nonzeroPx;
            }

            if((distanceToObject < lastDistanceToObject)) //check if current object is the closest
            {
                lastDistanceToObject = distanceToObject;

                int x_right = boundingBox.x + boundingBox.width;
                int x_left =  boundingBox.x;

                // distance between rights side of object and right wall in cm
                int distObjRightToWall = rightRange + HALF_CAR_WIDTH - (int)((x_right - 256)* (unsigned short)distanceToObject / 2560 * TAN_35 + 5);
                int distObjRightToWallPx = std::min((int)(((rightRange + HALF_CAR_WIDTH - 5) * 2560 / TAN_35 / (unsigned short)distanceToObject) - (x_right - 256)),511 - x_right);
                distObjRightToWallPx = std::max(distObjRightToWallPx, 1);

                rectFreeSpaceRight = cv::Rect(boundingBox.x + boundingBox.width + (int)(distObjRightToWallPx / 5), boundingBox.y, (int)(distObjRightToWallPx * 3 / 5), (int)(boundingBox.height * 2 / 3));
                rectFreeSpaceLeft = cv::Rect(std::max(0, boundingBox.x - 120), boundingBox.y, 100, (int)(boundingBox.height * 2 / 3));

                cv::Mat freeSpaceRight = im16(rectFreeSpaceRight);
                cv::Mat freeSpaceLeft = im16(rectFreeSpaceLeft);

                distanceFreeSpaceRight = 0;

                nonzeroPx = 0;
                for (int r = 0; r < freeSpaceRight.rows; ++r) {// calculates average of distance to right free space
                    for (int c = 0; c < freeSpaceRight.cols; ++c) {
                        if (freeSpaceRight.at<unsigned short>(r,c) != 0){
                            distanceFreeSpaceRight += freeSpaceRight.at<unsigned short>(r,c);
                            nonzeroPx++;
                        }
                    }
                }
                if(nonzeroPx != 0) {
                    distanceFreeSpaceRight = distanceFreeSpaceRight / nonzeroPx;
                }

                distanceFreeSpaceLeft = 0;

                nonzeroPx = 0;
                for (int r = 0; r < freeSpaceLeft.rows; ++r) {// calculates average of distance to left free space
                    for (int c = 0; c < freeSpaceLeft.cols; ++c) {
                        if (freeSpaceLeft.at<unsigned short>(r,c) != 0){
                            distanceFreeSpaceLeft += freeSpaceLeft.at<unsigned short>(r,c);
                            nonzeroPx++;
                        }
                    }
                }
                if(nonzeroPx != 0) {
                    distanceFreeSpaceLeft = distanceFreeSpaceLeft / nonzeroPx;
                }

                bool spaceFreeRight = distanceFreeSpaceRight > (unsigned short)(distanceToObject + 250);
                bool spaceFreeLeft = distanceFreeSpaceLeft > (unsigned short)(distanceToObject + 250);


                if(saveImages || publishImages){
                    // draw the rect
                    cv::rectangle(drawing, boundingBox, cv::Scalar(0,0,255), 1, 8, 0 );
                    cv::rectangle(drawing, rectFreeSpaceRight, cv::Scalar(0,255,0), 1, 8, 0 );
                    cv::rectangle(drawing, rectFreeSpaceLeft, cv::Scalar(0,255,0), 1, 8, 0 );
                }


                if(distanceToObject < 1700) {
                    if( distObjRightToWall > MIN_OBJ_DISTANCE_TO_WALL && spaceFreeRight) {//Space on the right is large enough and free
                        // pass on right
                        // returnDistance in m
                        returnDistance = std::max((double)0.2,((double)(distObjRightToWall/2 - HALF_CAR_WIDTH))/100); // was 0.05
                        avoidingState = PASS_RIGHT;
                    }
                    else if(spaceFreeLeft) {//Space on the right is to small
                        // pass on left
                        // distance between left side of object and right wall in cm
                        int distObjLeftToWall = rightRange + HALF_CAR_WIDTH - (int)((x_left - 256)* (unsigned short)distanceToObject / 2560 * TAN_35 + 5);

                        // returnDistance in m
                        returnDistance = std::min((double)1.5,((double) distObjLeftToWall)/100 + 0.2);
                        avoidingState = PASS_LEFT;
                    }
                    else {// no Free space
                        avoidingState = WALL;
                        returnDistance = DEFAULT_SET_VAL;
                    }
                }
            }
        }
    }


    if(saveImages || publishImages) { //save images for debugging

        cv::Mat adjustedImage;

        // generate timestamp
        time_t rawtime;
        struct tm * timeinfo;
        char buffer[80];
        time (&rawtime);
        timeinfo = localtime(&rawtime);
        strftime(buffer,sizeof(buffer),"%d-%m-%Y %I:%M:%S",timeinfo);
        std::string timestamp(buffer);

        if(adjustImages) {
            adjustedImage = (im16 + 50)*5;
        }
        else {
            adjustedImage = im16;
        }
        cv::Mat composedImage, concatinatedImage, concatinatedImageTop, concatinatedImageBottom;
        cv::Mat filteredImg_im8C3, threshedImage_im8C3, adjustedImage_im8C1, adjustedImage_im8C3, diffImage_im8_im8C3;

        adjustedImage.convertTo(adjustedImage_im8C1, CV_8UC1);
        cvtColor(adjustedImage_im8C1, adjustedImage_im8C3, CV_GRAY2RGB);

        cvtColor(diffImage_im8, diffImage_im8_im8C3, CV_GRAY2RGB);
        cvtColor(threshedImage, threshedImage_im8C3, CV_GRAY2RGB);

        cvtColor(filteredImg,filteredImg_im8C3,CV_GRAY2RGB);

        composedImage = cv::Mat::zeros(filteredImg.rows, filteredImg.cols, CV_8UC3);
        cv::scaleAdd(filteredImg_im8C3, 1, drawing, composedImage);

        if(avoidingState != NO_OBJECT) {

            std::ostringstream obstacleTextConverter;
            obstacleTextConverter << distanceToObject;
            std::string obstacleText = std::string("d=").append(obstacleTextConverter.str());
            std::ostringstream distanceFreeSpaceRightTextConverter;
            std::ostringstream distanceFreeSpaceLeftTextConverter;
            distanceFreeSpaceRightTextConverter << distanceFreeSpaceRight;
            distanceFreeSpaceLeftTextConverter << distanceFreeSpaceLeft;
            std::string freeSpaceRightText = std::string("d=").append(distanceFreeSpaceRightTextConverter.str());
            std::string freeSpaceLeftText = std::string("d=").append(distanceFreeSpaceLeftTextConverter.str());
            cv::putText(composedImage, obstacleText, cv::Point(boundingBox.x, boundingBox.y), 5, 0.5, cvScalar(0, 0, 255), 1, 8, false);
            cv::putText(composedImage, freeSpaceRightText, cv::Point(rectFreeSpaceRight.x, rectFreeSpaceRight.y), 5, 0.5, cvScalar(0, 255, 0), 1, 8, false);
            cv::putText(composedImage, freeSpaceLeftText, cv::Point(rectFreeSpaceLeft.x, rectFreeSpaceLeft.y), 5, 0.5, cvScalar(0, 255, 0), 1, 8, false);

        }

        cv::hconcat(adjustedImage_im8C3, diffImage_im8_im8C3, concatinatedImageTop);
        cv::hconcat(threshedImage_im8C3, composedImage, concatinatedImageBottom);
        cv::vconcat(concatinatedImageTop, concatinatedImageBottom, concatinatedImage);

        if(saveImages) {
            std::string fileNameDepthImage = std::string("/home/pses/catkin_ws/src/pses-turacers/detect_obstacle/Images/depth_image_").append(timestamp).append(".png");
            cv::imwrite(fileNameDepthImage,adjustedImage);
            std::string fileNameDiffImage = std::string("/home/pses/catkin_ws/src/pses-turacers/detect_obstacle/DiffImages/depth_image_").append(timestamp).append(".png");
            cv::imwrite(fileNameDiffImage,diffImage_im8);

            std::string fileNameBinImage = std::string("/home/pses/catkin_ws/src/pses-turacers/detect_obstacle/BinImages/image_").append(timestamp).append(".png");
            cv::imwrite(fileNameBinImage,filteredImg);
            std::string fileNameRectImage = std::string("/home/pses/catkin_ws/src/pses-turacers/detect_obstacle/RectImages/image_").append(timestamp).append(".png");
            cv::imwrite(fileNameRectImage,composedImage);
            std::string fileNameConcatinatedImage = std::string("/home/pses/catkin_ws/src/pses-turacers/detect_obstacle/ConcatImages/image_").append(timestamp).append(".png");
            cv::imwrite(fileNameConcatinatedImage,concatinatedImage);
        }
        if(publishImages){
            binaryImageVar = cv_bridge::CvImage(std_msgs::Header(), "bgr8", binImage_masked).toImageMsg();
            rectImageVar = cv_bridge::CvImage(std_msgs::Header(), "bgr8", composedImage).toImageMsg();
        }
    }
    imageReceived = false;
    return returnDistance;

}

// callback functions get called whenever a new message is availible in the input puffer
void uslCallback(sensor_msgs::Range::ConstPtr uslMsg, sensor_msgs::Range* usl)
{
    *usl = *uslMsg;
    if (usl->range == 0 ) {
        leftRange = 500;
    }
    else leftRange = usl->range * 100; // convert m to cm
}

void usrCallback(sensor_msgs::Range::ConstPtr usrMsg, sensor_msgs::Range* usr)
{
    *usr = *usrMsg;
    if (usr->range == 0 ) {
        rightRange = 500;
    }
    else rightRange = usr->range * 100; // convert m to cm
}

void kinectDepthCallback(const sensor_msgs::Image::ConstPtr& img){
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(*img, img->encoding);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }

    imCam = cv_ptr->image;
    imageReceived = true;
}


int main(int argc, char** argv)
{
    // init this node
    ros::init(argc, argv, "detect_obstacle_node");
    // get ros node handle
    ros::NodeHandle nh;

    // sensor message container
    sensor_msgs::Image::ConstPtr depth_img_ros;
    cv::Mat im16;

    // generate subscriber for sensor messages
    ros::Subscriber usrSub = nh.subscribe<sensor_msgs::Range>(
                "/uc_bridge/usr", 10, boost::bind(usrCallback, _1, &usr));
    ros::Subscriber uslSub = nh.subscribe<sensor_msgs::Range>(
                "/uc_bridge/usl", 10, boost::bind(uslCallback, _1, &usl));
    ros::Subscriber depthSub = nh.subscribe<sensor_msgs::Image>(
                "/kinect2/sd/image_depth", 1, &kinectDepthCallback);


    image_transport::ImageTransport it(nh);


    // generate control message publisher
    ros::Publisher recommendedDistance =
            nh.advertise<std_msgs::Float64>("wall_follow_obstacle/recommended_distance_msg", 1);
    ros::Publisher avoidingStatePub =
            nh.advertise<std_msgs::Int16>("wall_follow_obstacle/avoidung_state_msg", 1);

    ros::Publisher binaryImage =
            nh.advertise<sensor_msgs::Image>("wall_follow_obstacle/binary_image", 1);
    ros::Publisher rectImage =
            nh.advertise<sensor_msgs::Image>("wall_follow_obstacle/rect_image", 1);

    initializeValues();

    // Loop starts here:
    // loop rate value is set in Hz
    ros::Rate loop_rate(20);
    while (ros::ok())
    {

        //Obstacle Detection
        std_msgs::Float64 avoiding_wall_distance;

        if(imageReceived) {

            avoiding_wall_distance.data = detectObstacle(imCam);

            // publish command messages on their topics
            avoidingStatePub.publish(avoidingState);
            recommendedDistance.publish(avoiding_wall_distance);
            if(publishImages) {
                binaryImage.publish(binaryImageVar);
                rectImage.publish(rectImageVar);
            }

            imageReceived = false;
        }

        // clear input/output buffers
        ros::spinOnce();
        // this is needed to ensure a const. loop rate
        loop_rate.sleep();
    }

    ros::spin();
}
