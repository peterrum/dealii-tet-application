SET(TARGET_SRC main.cc)
ADD_EXECUTABLE( "poisson_hp" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("poisson_hp")
TARGET_LINK_LIBRARIES("poisson_hp" "ceed")