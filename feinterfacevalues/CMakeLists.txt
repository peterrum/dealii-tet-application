SET(TARGET_SRC main.cc)
ADD_EXECUTABLE( "feinterfacevalues" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("feinterfacevalues")
TARGET_LINK_LIBRARIES("feinterfacevalues" "ceed")