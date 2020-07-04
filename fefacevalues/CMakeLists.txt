SET(TARGET_SRC main.cc)
ADD_EXECUTABLE( "fefacevalues" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("fefacevalues")
TARGET_LINK_LIBRARIES("fefacevalues" "ceed")