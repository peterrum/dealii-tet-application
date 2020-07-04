SET(TARGET_SRC main.cc)
ADD_EXECUTABLE( "gridinout" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("gridinout")
TARGET_LINK_LIBRARIES("gridinout" "ceed")