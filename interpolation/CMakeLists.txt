SET(TARGET_SRC main.cc)
ADD_EXECUTABLE( "interpolation" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("interpolation")
TARGET_LINK_LIBRARIES("interpolation" "ceed")
