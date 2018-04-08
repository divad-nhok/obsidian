//!
//! Contains the implementation of an adaptive proposal function
//!
//! \file infer/adaptive.cpp
//! \author David Kohn
//! \date 2017
//! \license Affero General Public License version 3 or later
//! \copyright (c) 2014, NICTA
//!

#include "adaptive.hpp"

namespace stateline
{
  namespace mcmc
  {
    
    //! A function to bounce the MCMC proposal off the hard boundaries.
    //! This allows the proposal to always move around without getting stuck at
    //! 'walls'
    //! 
    //! \param val The proposed value
    //! \param min The minimum bound of theta 
    //! \param max The maximum bound of theta 
    //!  \returns The new bounced theta definitely in the bounds
    //!
    Eigen::VectorXd bouncyBounds(const Eigen::VectorXd& val,const Eigen::VectorXd& min, const Eigen::VectorXd& max)
    { 
      Eigen::VectorXd delta = max - min;
      Eigen::VectorXd result = val;
      Eigen::Matrix<bool, Eigen::Dynamic, 1> tooBig = (val.array() > max.array());
      Eigen::Matrix<bool, Eigen::Dynamic, 1> tooSmall = (val.array() < min.array());
      for (uint i=0; i< result.size(); i++)
      {
        bool big = tooBig(i);
        bool small = tooSmall(i);
        if (big)
        {
          double overstep = val(i)-max(i);
          int nSteps = (int)(overstep /  delta(i));
          double stillToGo = overstep - nSteps*delta(i);
          if (nSteps % 2 == 0)
            result(i) = max(i) - stillToGo;
          else
            result(i) = min(i) + stillToGo;
        }
        if (small)
        {
          double understep = min(i) - val(i);
          int nSteps = (int)(understep / delta(i));
          double stillToGo = understep - nSteps*delta(i);
          if (nSteps % 2 == 0)
            result(i) = min(i) + stillToGo;
          else
            result(i) = max(i) - stillToGo;
        }
      }
      return result;
    }
    
    //! An adaptive Gaussian proposal function. It randomly varies each value in
    //! the state according to a Gaussian distribution whose variance changes
    //! depending on the acceptance ratio of a chain. It also bounces of the
    //! walls of the hard boundaries given so as not to get stuck in corners.
    //! 
    //! \param state The current state of the chain
    //! \param sigma The standard deviation of the distribution (step size of the proposal)
    //! \param qcov Dummy parameter added by RS 2018/03/09 to match calling structure
    //! \param min The minimum bound of theta 
    //! \param max The maximum bound of theta 
    //! \returns The new proposed theta
    //!
    Eigen::VectorXd adaptiveGaussianProposal(const Eigen::VectorXd &state, double sigma, 
        const Eigen::MatrixXd& qcov, const Eigen::VectorXd& min, const Eigen::VectorXd& max)
    {
      // Random number generators
      static std::random_device rd;
      static std::mt19937 generator(rd());
      static std::normal_distribution<> rand; // Standard normal

      // Vary each paramater according to a Gaussian distribution
      Eigen::VectorXd proposal(state.rows());
      for (int i = 0; i < proposal.rows(); i++)
        proposal(i) = state(i) + rand(generator) * sigma;

      //return bouncyBounds(proposal, min, max);
      return(proposal);
    };
    
    //! RS 2018/03/09:  A multivariate Gaussian proposal function.
    //! It turns out random walk proposals made from a Gaussian with correlated
    //! components won't satisfy detailed balance if made across a reflection
    //! boundary, so it's important that we NOT do this, and instead just set
    //! the world prior probability to zero in order to auto-reject the state.
    //!
    //! \param state The current state of the chain
    //! \param sigma Scaling parameter (step size) to apply to chain covariance
    //! \param qcov Covariance of a multivariate Gaussian with ~unit diagonal
    //! \param min The minimum bound of theta 
    //! \param max The maximum bound of theta 
    //! \returns The new proposed theta
    //!
    Eigen::VectorXd multiGaussianProposal(const Eigen::VectorXd &state, double sigma, 
        const Eigen::MatrixXd& qcov, const Eigen::VectorXd& min, const Eigen::VectorXd& max)
    {
      // Random number generators
      static std::random_device rd;
      static std::mt19937 generator(rd());
      static std::normal_distribution<> rand; // Standard normal

      // Draw from a multivariate Gaussian
      Eigen::VectorXd zero_mean = 0.0*state;

      obsidian::distrib::MultiGaussian q(zero_mean, qcov);
      return state + sigma*obsidian::distrib::drawValues(q, generator);
    };
    
  }
}
