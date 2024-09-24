#include <pluginlib/class_list_macros.h>
#include <tf/tf.h>
#include <cmath>
#include "glplan.h"

// Register this planner as a BaseGlobalPlanner plugin
PLUGINLIB_EXPORT_CLASS(global_planner::GlobalPlanner, nav_core::BaseGlobalPlanner)

namespace global_planner {

GlobalPlanner::GlobalPlanner() {}

GlobalPlanner::GlobalPlanner(std::string name, costmap_2d::Costmap2DROS* costmap_ros) {
    initialize(name, costmap_ros);
}

void GlobalPlanner::initialize(std::string name, costmap_2d::Costmap2DROS* costmap_ros) {
    path_sub_ = nh_.subscribe("/path_request", 1000, &GlobalPlanner::pathCallback, this);
}

bool GlobalPlanner::makePlan(const geometry_msgs::PoseStamped& start, const geometry_msgs::PoseStamped& goal, std::vector<geometry_msgs::PoseStamped>& plan) {
    if (path_poses.empty()) {
        ROS_WARN("No path poses received yet.");
        return false;
    }

    plan.push_back(path_poses.front());
    for (const auto& pose : path_poses) {
        plan.push_back(pose);
    }
    plan.push_back(path_poses.back());
    return true;
}

void GlobalPlanner::pathCallback(const nav_msgs::Path::ConstPtr& msg) {
    ROS_INFO("Received a path with %lu poses", msg->poses.size());
    path_poses.clear();
    for (const auto& pose : msg->poses) {
        path_poses.push_back(pose);
    }
}

}  // namespace global_planner
