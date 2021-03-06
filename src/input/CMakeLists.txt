# Copyright (c) 2014, NICTA. 
# Affero General Public License version 3 or later
# See the COPYRIGHT file.

# Authors: Lachlan McCalman
# Date: 2014 

ADD_LIBRARY(input input.cpp
                  mcmc.cpp
                  global.cpp
                  world.cpp
                  gravity.cpp
                  magnetism.cpp
                  mt.cpp
                  seismic.cpp
                  contactpoint.cpp
                  thermal.cpp
                  fieldobs.cpp)

SET(TESTLIBS ${obsidianAlgoLibraries} ${obsidianServerLibraries} ${obsidianServerLibraries} ${obsidianBaseLibraries})

ADD_EXECUTABLE(test-input testinput.cpp)
TARGET_LINK_LIBRARIES(test-input ${TESTLIBS} )
REGISTER_UNIT_TESTS(test-input)

ADD_EXECUTABLE(test-input-world testworld.cpp)
TARGET_LINK_LIBRARIES(test-input-world ${TESTLIBS} )
REGISTER_UNIT_TESTS(test-input-world)

ADD_EXECUTABLE(test-input-gravity testgravity.cpp)
TARGET_LINK_LIBRARIES(test-input-gravity ${TESTLIBS} )
REGISTER_UNIT_TESTS(test-input-gravity)

ADD_EXECUTABLE(test-input-magnetism testmagnetism.cpp)
TARGET_LINK_LIBRARIES(test-input-magnetism ${TESTLIBS} )
REGISTER_UNIT_TESTS(test-input-magnetism)

ADD_EXECUTABLE(test-input-mt testmt.cpp)
TARGET_LINK_LIBRARIES(test-input-mt ${TESTLIBS} )
REGISTER_UNIT_TESTS(test-input-mt)

ADD_EXECUTABLE(test-input-seismic testseismic.cpp)
TARGET_LINK_LIBRARIES(test-input-seismic ${TESTLIBS} )
REGISTER_UNIT_TESTS(test-input-seismic)

ADD_EXECUTABLE(test-input-contactpoint testcontactpoint.cpp)
TARGET_LINK_LIBRARIES(test-input-contactpoint ${TESTLIBS} )
REGISTER_UNIT_TESTS(test-input-contactpoint)

ADD_EXECUTABLE(test-input-thermal testthermal.cpp)
TARGET_LINK_LIBRARIES(test-input-thermal ${TESTLIBS} )
REGISTER_UNIT_TESTS(test-input-thermal)

ADD_EXECUTABLE(test-input-fieldobs testfieldobs.cpp)
TARGET_LINK_LIBRARIES(test-input-fieldobs ${TESTLIBS} )
REGISTER_UNIT_TESTS(test-input-fieldobs)
