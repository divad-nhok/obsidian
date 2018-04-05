#include "prior/world.hpp"
#include <Eigen/Core>
#include <glog/logging.h>
#include <cassert>
#include <limits>
#include <iostream>

namespace obsidian
{
  namespace prior
  {
    uint prod(std::pair<uint, uint> in)
    {
      return in.first * in.second;
    }

    // get the number of partition elements
    uint WorldParamsPrior::size()
    {
      // Just get the size of the objects
      uint size = 0;
      uint nLayers = propertyPrior.size();
      CHECK_EQ(ctrlptPrior.size(), nLayers);
      for (uint i = 0; i < nLayers; i++)
      {
        uint sizeRock = prod(propertyPrior[i].shape);
        for (uint j = 0; j < sizeRock; j++)
          size += propMasks[i](j);
        for (uint j = 0; j < ctrlptPrior[i].shape.first; j++)
          for (uint k = 0; k < ctrlptPrior[i].shape.second; k++)
          {
            size += ctrlptMasks[i](j, k);
          }
      }
      return size;
    }

    double whiten(double x, double mu, double sigmaSq, double min, double max)
    {
      double sd = std::sqrt(sigmaSq);
      double range = max - min;
      double scale = std::min(sd, range / 5.0);
      return (x - mu) / scale;
    }

    double unwhiten(double x, double mu, double sigmaSq, double min, double max)
    {
      double sd = std::sqrt(sigmaSq);
      double range = max - min;
      double scale = std::min(sd, range / 5.0);
      return (x * scale) + mu;
    }

    Eigen::VectorXd WorldParamsPrior::thetaMinBound()
    {
      return thetaMin;
    }

    Eigen::VectorXd WorldParamsPrior::thetaMaxBound()
    {
      return thetaMax;
    }

    // used by sample... almost the same as size()
    Eigen::VectorXd WorldParamsPrior::deconstruct(const WorldParams& params)
    {
      uint nLayers = propertyPrior.size();
      CHECK_EQ(nLayers, params.rockProperties.size());
      CHECK_EQ(nLayers, params.controlPoints.size());
      std::vector<double> theta;
      for (uint i = 0; i < nLayers; i++)
      {
        uint sizeRock = prod(propertyPrior[i].shape);
        CHECK_EQ(sizeRock, params.rockProperties[i].rows());

        for (uint j = 0; j < sizeRock; j++)
        {
          if (propMasks[i](j)) // we're using this property
            theta.push_back(
                whiten(params.rockProperties[i](j), propertyPrior[i].mu[j], propertyPrior[i].sigma(j, j), propMins[i](j), propMaxs[i](j)));
        }
      }
      for (uint i = 0; i < nLayers; i++)
      {
        uint sizeCtrl = prod(ctrlptPrior[i].shape);
        CHECK_EQ(sizeCtrl, params.controlPoints[i].rows() * params.controlPoints[i].cols());
        for (uint j = 0; j < ctrlptPrior[i].shape.first; j++)
          for (uint k = 0; k < ctrlptPrior[i].shape.second; k++)
          {
            auto ej = j * ctrlptPrior[i].shape.second + k;
            //CHECK(ej>=0);
            //CHECK(ej < sizeCtrl);
            if (ctrlptMasks[i](j, k)) // we're using this control point
              theta.push_back(
                  whiten(params.controlPoints[i](j, k), ctrlptPrior[i].mu(ej), ctrlptPrior[i].sigma(ej, ej), ctrlptMins[i](j, k),
                         ctrlptMaxs[i](j, k)));
          }
      }

      // std::vector to Eigen vector
      Eigen::VectorXd vTheta(theta.size());
      for (uint i = 0; i < theta.size(); i++)
        vTheta(i) = theta[i];
      return vTheta;
    }

    // build a WorldParams object from a flat vector
    WorldParams WorldParamsPrior::reconstruct(const Eigen::VectorXd & theta)
    {
      uint nLayers = propertyPrior.size();
      CHECK_EQ(ctrlptPrior.size(), nLayers);
      std::vector<Eigen::VectorXd> propTheta(nLayers);
      std::vector<Eigen::MatrixXd> geomTheta(nLayers);
      uint count = 0;
      for (uint i = 0; i < nLayers; i++)
      {
        uint sizeRock = prod(propertyPrior[i].shape);
        Eigen::VectorXd rockValues(sizeRock);
        for (uint j = 0; j < sizeRock; j++)
        {
          if (propMasks[i](j)) // we're using this property
          {
            rockValues(j) = unwhiten(theta(count), propertyPrior[i].mu(j), propertyPrior[i].sigma(j, j), propMins[i](j), propMaxs[i](j));
            count++;
          } else
          {
            rockValues(j) = propertyPrior[i].mu(j);
          }
        }
        propTheta[i] = rockValues;
      }

      for (uint i = 0; i < nLayers; i++)
      {
        std::pair<uint, uint> layerShape = ctrlptPrior[i].shape;
        Eigen::MatrixXd geomValues(layerShape.first, layerShape.second);
        for (uint j = 0; j < layerShape.first; j++)
          for (uint k = 0; k < layerShape.second; k++)
          {
            auto ej = j * ctrlptPrior[i].shape.second + k;
            if (ctrlptMasks[i](j, k)) // we're using this control point
            {
              geomValues(j, k) = unwhiten(theta(count), ctrlptPrior[i].mu(ej), ctrlptPrior[i].sigma(ej, ej), ctrlptMins[i](j, k),
                                          ctrlptMaxs[i](j, k));
              count++;
            } else
            {
              // TODO check I'm indexing correctly here
              geomValues(j, k) = ctrlptPrior[i].mu(j * layerShape.second + k);
            }
          }
        geomTheta[i] = geomValues;
      }

      WorldParams wParams = { propTheta, geomTheta };
      return wParams;
    }

    Eigen::VectorXd WorldParamsPrior::sample(std::mt19937 &gen)
    {
      std::vector<bool> uniformFlags;
      for (auto i : classes)
        uniformFlags.push_back(i == BoundaryClass::Warped);
      
      // RS 2018/03/22:  Keep drawing thetas until we get a parameter vector
      // with all the components within the given hard bounds.  Try at most
      // max_inbound_sample_tries times to avoid infinite loops.
      uint max_inbounds_sample_tries = 1000;
      class ThetaBoundsError { };
      uint i;
      Eigen::VectorXd theta_draw;

      for (i = 0; i < max_inbounds_sample_tries; i++)
      {
        WorldParams prior_draw = { distrib::drawVectorFrom(propertyPrior, gen, propMins, propMaxs),
                                   distrib::drawFrom(ctrlptPrior, gen, ctrlptMins, ctrlptMaxs, uniformFlags) };
        theta_draw = deconstruct(prior_draw);
        bool is_good_draw = 1;
        for (uint i = 0; i < theta_draw.size(); i++)
          if (theta_draw[i] < thetaMin[i] or theta_draw[i] > thetaMax[i])
            is_good_draw = 0;
        if (is_good_draw) return theta_draw;
      }

      // We didn't get a good draw, so throw an exception
      if (i == max_inbounds_sample_tries)
      {
        std::cout << "FATAL: couldn't draw valid prior vector in" << max_inbounds_sample_tries << "tries";
        throw ThetaBoundsError { };
      }
      return theta_draw;
    }

    // Evaluate log likelihood of theta under this prior
    double WorldParamsPrior::evaluatePDF(const Eigen::VectorXd& theta)
    {

      uint nLayers = propertyPrior.size();
      CHECK_EQ(ctrlptPrior.size(), nLayers);
      // Rebuild the vector into an object
      WorldParams wParams = reconstruct(theta);

      // RS 2018/03/22:  If any of the thetas are out of bounds, this world
      // has basically zero probability.  This is a blunt instrument, but it
      // is needed for any MCMC proposal that isn't an independent Metropolis
      // random walk to obey detailed balance at the parameter bounds.
      // I'm not sure if we're IEEE 754 compliant but otherwise we'll have to
      // pick some arbitrary large negative number which I could imagine
      // giving unexpected results.  We'll also need range checks to ensure
      // we don't *start* chains in regions of zero probability.
      static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
      for (uint i = 0; i < theta.size(); i++)
        if (theta[i] < thetaMin[i] or theta[i] > thetaMax[i])
          return -std::numeric_limits<double>::infinity();

      // Now we go and check each of these against their prior
      double logPDF = 0.0;
      for (uint i = 0; i < nLayers; i++)
      {
        logPDF += distrib::logPDF(wParams.rockProperties[i], propertyPrior[i], propMins[i], propMaxs[i]);
        bool isUniform = classes[i] == BoundaryClass::Warped;
        if (isUniform)
        {
          logPDF += distrib::uniformLogPDF(wParams.controlPoints[i], ctrlptPrior[i], ctrlptMins[i], ctrlptMaxs[i]);
        } else
        {
          logPDF += distrib::logPDF(wParams.controlPoints[i], ctrlptPrior[i], ctrlptMins[i], ctrlptMaxs[i]);
        }
      }
      return logPDF;
    }

  }
} // namespace obsidian

