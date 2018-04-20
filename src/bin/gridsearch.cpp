//!
//! Output the results of an obsidian run in open format.
//!
//! \file gridsearch.cpp
//! \author David Kohn
//! \date 18 April, 2018
//! \license Affero General Public License version 3 or later
//! \copyright (c) 2018
//!

// Standard Library
#include <random>
#include <thread>
#include <chrono>
#include <numeric>
// Prerequisites
#include <glog/logging.h>
#include <boost/program_options.hpp>
// Project
#include "app/console.hpp"
#include "app/settings.hpp"
#include "input/input.hpp"
#include "comms/delegator.hpp"
#include "comms/requester.hpp"
#include "serial/serial.hpp"
#include "app/asyncdelegator.hpp"
#include "app/signal.hpp"
#include "infer/mcmc.hpp"
#include "infer/metropolis.hpp"
#include "infer/adaptive.hpp"
#include "fwdmodel/global.hpp"
#include "datatype/sensors.hpp"
#include "io/npy.hpp"
#include "io/dumpnpz.hpp"
#include "likelihood/likelihood.hpp"
#include "world/property.hpp"
#include "detail.hpp"

using namespace obsidian;
using namespace stateline;

namespace ph = std::placeholders;

// Command line options specific to obsidian
po::options_description commandLineOptions()
{
  po::options_description cmdLine("Delegator Command Line Options");
  cmdLine.add_options() //
    ("configfile,c", po::value<std::string>()->default_value("obsidian_config"), "configuration file") //
    ("inputfile,i", po::value<std::string>()->default_value("input.obsidian"), "input file") //
    ("outputfile,o", po::value<std::string>()->default_value("gridsearch.npz"), "output file") //
    ("pickaxefile,p", po::value<std::string>()->default_value("output.npz"), "output file from pickaxe") //
    ("fwdmodel,f", po::bool_switch()->default_value(true), "Also compute forward models") //
    ("recover,r", po::bool_switch()->default_value(true), "recovery (deprecated)");
  return cmdLine;
}

int main(int ac, char* av[])
{
  init::initialiseSignalHandler();
  // Get the settings from the command line
  auto vm = init::initProgramOptions(ac, av, commandLineOptions());

  // Set up a random generator here
  std::random_device rd;
  std::mt19937 gen(rd());

  readConfigFile(vm["configfile"].as<std::string>(), vm);
  readInputFile(vm["inputfile"].as<std::string>(), vm);
  std::set<ForwardModel> sensorsEnabled = parseSensorsEnabled(vm);
  MCMCSettings mcmcSettings = parseMCMCSettings(vm);
  DBSettings dbSettings = parseDBSettings(vm);
  dbSettings.recover = true;
  GlobalPrior prior = parsePrior<GlobalPrior>(vm, sensorsEnabled);
  
  // Stuff needed for forward modelling
  GlobalSpec globalSpec = parseSpec<GlobalSpec>(vm, sensorsEnabled);
  std::vector<world::InterpolatorSpec> boundaryInterp = world::worldspec2Interp(globalSpec.world);
  GlobalResults realResults = loadResults(globalSpec.world, vm, sensorsEnabled);
  GlobalCache cache = fwd::generateGlobalCache(boundaryInterp, globalSpec, sensorsEnabled);
  std::vector<uint> sensorId;
  std::vector<std::string> sensorSpecData;
  std::vector<std::string> sensorReadings;

  std::string filename = vm["outputfile"].as<std::string>();

  Eigen::MatrixXd thetas = io::readNPZ(vm["pickaxefile"].as<std::string>(), "thetas");
  uint nsamples = thetas.rows();
  LOG(INFO) << "nsamples: " << nsamples;

  std::vector<WorldParams> dumpParams;
  std::vector<double> dumpPrior;
  std::vector<Eigen::VectorXd> dumpTheta;

  // Forward modelling stuff
  bool computeForwardModels = vm["fwdmodel"].as<bool>();
  std::vector<GlobalResults> dumpResults;
  std::vector<std::vector<double>> dumpLikelihood;

  io::NpzWriter writer(filename);
  bool quit = false;
  for (uint idx = 0; idx < nsamples; idx++)
  {
    LOG(INFO) << "idx: " << idx;
    if(global::interruptedBySignal)
    {
      quit = true;
      break;
    }
    // Reconstruct world model
    Eigen::VectorXd theta = thetas.row(idx);
    GlobalParams params = prior.reconstruct(theta);
    // prior
    double priorLikelihood = prior.evaluate(theta);
    dumpPrior.push_back(priorLikelihood);
    dumpParams.push_back(params.world);
    dumpTheta.push_back(theta);

    if (computeForwardModels)
    {
      GlobalResults results = fwd::forwardModelAll(globalSpec, cache, params, sensorsEnabled);
      dumpResults.push_back(results);
      std::vector<double> likelihoods = lh::likelihoodAll(results, realResults, globalSpec, sensorsEnabled);
      dumpLikelihood.push_back(likelihoods); 
    }

    if (quit)
      break;
  }

  dumpParamsNPZ(writer, dumpParams);
  dumpPriorNPZ(writer, dumpPrior);
  dumpThetaNPZ(writer, dumpTheta);

  if (computeForwardModels)
  {
    dumpResultsNPZ(writer, dumpResults);
    dumpLikelihoodNPZ(writer, dumpLikelihood);
  }

  return 0;
}
