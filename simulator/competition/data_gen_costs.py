# Use acceleration vector or component for upper bound check
ACCEL_VECTOR = True

# Accel upper bounds p_shape * v^2 + p_intercept
PARABOLA_INTERCEPT = 2.5
PARABOLA_SHAPE = -0.675  ## this should be negative
VEL_LIMIT_COST = 2.0

########## ALL VALUES HERE REGARdING COSTS SHOULD BE POSITIVE ########
ACCEL_COST = 50000
VEL_COST = 50000

CRASH_COST = 1000000
SKIPPED_WAYPOINT_COST = 100
TRACKING_COST = 4
TIME_COST = 1

ACCEL_RANGE_COST = 0
