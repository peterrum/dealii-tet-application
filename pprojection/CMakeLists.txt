SET(TARGET_SRC main.cc)
ADD_EXECUTABLE( "pprojection" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("pprojection")
TARGET_LINK_LIBRARIES("pprojection" "ceed")
