SET(TARGET_SRC main.cc)
ADD_EXECUTABLE( "step18" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("step18")
TARGET_LINK_LIBRARIES("step18" "ceed")
