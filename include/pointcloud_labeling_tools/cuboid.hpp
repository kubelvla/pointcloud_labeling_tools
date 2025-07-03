#pragma once

#include <Eigen/Dense>

struct Cuboid {
    Eigen::Vector3f center;
    Eigen::Vector3f extent;
    Eigen::Matrix3f rotation;
    int id;
};

inline bool isInsideCuboid(const Eigen::Vector3f& pt, const Cuboid& box) {
    Eigen::Vector3f local = box.rotation.transpose() * (pt - box.center);
    return std::abs(local.x()) <= box.extent.x() / 2 &&
           std::abs(local.y()) <= box.extent.y() / 2 &&
           std::abs(local.z()) <= box.extent.z() / 2;
}