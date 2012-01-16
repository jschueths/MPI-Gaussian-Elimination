###############################################################################
# -->Makefile<--
###############################################################################

# Author :  Joshua McCarville-Schueths

###############################################################################

###############################################################################
# This makefile will build an executable for the project.
###############################################################################

.PHONY: all clean

# The following 2 lines only work with gnu make.
# It's much nicer than having to list them out,
# and less error prone.
SOURCES = $(wildcard *.cpp)
HEADERS = $(wildcard *.h)

# With Sun's make it has to be done like this, instead of
# using wildcards.
# Well, I haven't figured out another way yet.
#SOURCES = signal.cpp tokentype.cpp token.cpp tokenlist.cpp driver.cpp
#HEADERS = signal.h tokentype.h token.h tokenlist.h

# Looks like it can be done like this, but won't work for gmake.
#SOURCES:sh = ls *.cpp
#HEADERS:sh = ls *.h

OBJECTS = $(SOURCES:%.cpp=%.o)

default: gaussian

%.o: %.cpp
	@echo "Compiling $<"
	@mpic++ -c $< -o $@

gaussian: $(OBJECTS)
	@echo "Building $@"
	@mpic++ $(OBJECTS) -o $@
	@echo ""
	@echo "Everything worked :-) "
	@echo ""

clean:
	-@rm -f core
	-@rm -f gaussian
	-@rm -f $(OBJECTS)
	-@rm -f depend
	-@rm -f *~

# Automatically generate dependencies and include them in Makefile
depend: $(SOURCES) $(HEADERS)
	@echo "Generating dependencies"
	@$(CXX) -MM *.cpp > $@


-include depend
# Put a dash in front of include when using gnu make.
# It stops gmake from warning us that the file
# doesn't exist yet, even though it will be properly
# made and included when all is said and done.
