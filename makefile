.ONESHELL:
.SECONDEXPANSION:

APP := render

#the default target
all: monkey.svg torus.svg

PKG := cairo libavformat libavcodec libavutil libswscale
PKG_CONFIG_FLAGS := $(shell pkg-config $(PKG) --cflags)
PKG_CONFIG_LD := $(shell pkg-config $(PKG) --libs)

CXXFLAGS += -std=c++14 -Iinclude -fno-rtti -fno-exceptions -Wall $(PKG_CONFIG_FLAGS)
ifdef DEBUG
CXXFLAGS += -D_GLIBCXX_DEBUG -g
else
CXXFLAGS += -ffunction-sections -fdata-sections -O3
endif

render.o: %.o : %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -c

render.cpp.format : %.format : % 
	clang-format -style="{BasedOnStyle: Chromium,ColumnLimit: 160}" -i $<
	echo > $@
	
LDFLAGS += -lm
LDFLAGS += $(PKG_CONFIG_LD)
ifndef DEBUG
LDFLAGS += -Wl,--gc-sections
endif

$(APP) : render.o
	$(CXX) $(LDFLAGS) $^ -o $@

monkey.svg : $(APP) monkey.stl 
	./$^ $@ -m "0 2 0"

torus.svg : $(APP) torus.stl
	./$^ $@ --rotate_x ".61" -m "0 2 0"

format: render.cpp.format
.PHONY: format

clean:
	rm -rf $(APP) monkey.svg torus.svg render.o render.cpp.format
.PHONY: clean
