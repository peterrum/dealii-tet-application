SET(TARGET_SRC main.cc)
ADD_EXECUTABLE( "lethe" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("lethe")
TARGET_LINK_LIBRARIES("lethe" "ceed")