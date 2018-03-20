//!
//! Contains the implementation of an adaptive proposal function
//!
//! \file infer/adaptive.hpp
//! \author Darren Shen
//! \date 2014
//! \license Affero General Public License version 3 or later
//! \copyright (c) 2014, NICTA
//!

#pragma once

#include <random>
#include <functional>
#include <Eigen/Core>
#include "distrib/multigaussian.hpp"

namespace stateline
{
  namespace mcmc
  {
    Eigen::VectorXd bouncyBounds(const Eigen::VectorXd& val,const Eigen::VectorXd& min, const Eigen::VectorXd& max);
    
    Eigen::VectorXd adaptiveGaussianProposal(const Eigen::VectorXd &state, double sigma, 
        const Eigen::MatrixXd& qcov, const Eigen::VectorXd& min, const Eigen::VectorXd& max);

    Eigen::VectorXd multiGaussianProposal(const Eigen::VectorXd &state, double sigma, 
        const Eigen::MatrixXd& qcov, const Eigen::VectorXd& min, const Eigen::VectorXd& max);
    
  }
}
