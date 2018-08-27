// Copyright (c) 2014, NICTA.
// This file is licensed under the General Public License version 3 or later.
// See the COPYRIGHT file.

/**
 * Contains the basic GDF datatypes for communication between forward models.
 *
 * @file testcontactpoint.hpp
 * @author Nahid Akbar
 * @date 2014
 */

#include "serial/fieldobs.hpp"
#include "test/serial.hpp"
#include "test/fieldobs.hpp"

namespace obsidian
{
TEST_F(Serialise, testSpec)
{
  generateVariations<FieldObsSpec>(test<FieldObsSpec>);
}

TEST_F(Serialise, testParams)
{
  generateVariations<FieldObsParams>(test<FieldObsParams>);
}

TEST_F(Serialise, testResults)
{
  generateVariations<FieldObsResults>(test<FieldObsResults>);
}
}
