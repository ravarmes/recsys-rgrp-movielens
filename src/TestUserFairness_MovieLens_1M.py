from RecSys import RecSys
from UserFairness import Polarization
from UserFairness import IndividualLossVariance
from UserFairness import GroupLossVariance


# reading data from 3883 movies and 6040 users 
Data_path = 'Data/MovieLens-1M'
n_users=  300
n_items= 1000
top_users = True # True: to use users with more ratings; False: otherwise
top_items = False # True: to use movies with more ratings; False: otherwise

# recommendation algorithm
algorithm = 'RecSysALS'

# parameters for calculating fairness measures
l = 5
theta = 3
k = 3

recsys = RecSys(n_users, n_items, top_users, top_items, l, theta, k)

X, users_info, items_info = recsys.read_movielens_1M(n_users, n_items, top_users, top_items, data_dir = Data_path) # returns matrix of ratings with n_users rows and n_items columns
omega = ~X.isnull() # matrix X with True in cells with evaluations and False in cells not rated

X_est = recsys.compute_X_est(X, algorithm) # RecSysALS or RecSysKNN or RecSysNMF or RecSysExampleAntidoteData20Items

print("\n\n------------ SOCIAL OBJECTIVE FUNCTIONS ------------")

# To capture polarization, we seek to measure the extent to which the user ratings disagree
polarization = Polarization()
Rpol = polarization.evaluate(X_est)
print("Polarization (Rpol):", Rpol)

# Individual fairness. For each user i, the loss of user i, is  the mean squared estimation error over known ratings of user i
ilv = IndividualLossVariance(X, omega, 1) #axis = 1 (0 rows e 1 columns)
Rindv = ilv.evaluate(X_est)
print("Individual Loss Variance (Rindv):", Rindv)

# Group fairness. Let I be the set of all users/items and G = {G1 . . . ,Gg} be a partition of users/items into g groups. 
# The loss of group i as the mean squared estimation error over all known ratings in group i

# G group: identifying the groups (NR: users grouped by number of ratings for available items)
# advantaged group: 5% users with the highest number of item ratings
# disadvantaged group: 95% users with the lowest number of item ratings
list_users = X_est.index.tolist()
advantaged_group = list_users[0:15]
disadvantaged_group = list_users[15:300]
G1 = {1: advantaged_group, 2: disadvantaged_group}

glv = GroupLossVariance(X, omega, G1, 1) #axis = 1 (0 rows e 1 columns)
RgrpNR = glv.evaluate(X_est)
print("Group Loss Variance (Rgrp NR):", RgrpNR)
print("Group Loss Variance (Rgrp NR): %.7f" %RgrpNR)


# G group: identifying the groups (IU: individual unfairness - the variance of the user losses)
# The configuration of groups was based in the hierarchical clustering (tree clustering - dendrogram)
# Clusters 1, 2 and 3
G2 = {1: [48, 53, 123, 149, 173, 195, 245, 308, 319, 331, 424, 482, 509, 524, 660, 678, 692, 699, 710, 721, 839, 855, 877, 889, 984, 1015, 1050, 1068, 1088, 1112, 1117, 1120, 1137, 1181, 1194, 1203, 1224, 1264, 1266, 1274, 1285, 1298, 1317, 1383, 1422, 1425, 1449, 1451, 1465, 1496, 1579, 1605, 1613, 1635, 1647, 1676, 1698, 1733, 1737, 1748, 1780, 1835, 1837, 1880, 1884, 1889, 1897, 1899, 1926, 1958, 1980, 1988, 2010, 2012, 2030, 2116, 2304, 2544, 2793, 2857, 2887, 2934, 2962, 2986, 3018, 3029, 3067, 3118, 3163, 3224, 3272, 3292, 3336, 3389, 3391, 3471, 3483, 3539, 3618, 3626, 3675, 3681, 3693, 3705, 3724, 3768, 3778, 3821, 3824, 3934, 4033, 4054, 4064, 4085, 4089, 4140, 4169, 4238, 4277, 4312, 4345, 4425, 4447, 4448, 4482, 4543, 4579, 4725, 4728, 4732, 4802, 4808, 4867, 4957, 5011, 5054, 5100, 5220, 5256, 5312, 5333, 5394, 5504, 5511, 5536, 5627, 5636, 5675, 5747, 5812, 5831, 5916, 6016, 6036], 2: [148, 202, 216, 302, 329, 352, 411, 438, 528, 531, 533, 543, 549, 550, 731, 869, 881, 1010, 1019, 1125, 1150, 1207, 1246, 1303, 1333, 1354, 1447, 1448, 1632, 1671, 1675, 1680, 1741, 1749, 1764, 1912, 1920, 1941, 1943, 2063, 2077, 2092, 2106, 2109, 2124, 2181, 2453, 2507, 2529, 2777, 2820, 2878, 2909, 3032, 3280, 3285, 3311, 3320, 3401, 3410, 3462, 3491, 3507, 3519, 3562, 3648, 3650, 3683, 3792, 3808, 3823, 3829, 3834, 3841, 3884, 3885, 3929, 3942, 3999, 4016, 4021, 4041, 4186, 4227, 4305, 4344, 4386, 4411, 4508, 4510, 4578, 4647, 4682, 5015, 5026, 5046, 5107, 5111, 5306, 5367, 5387, 5433, 5493, 5501, 5530, 5550, 5614, 5643, 5682, 5759, 5788, 5880, 5888, 5996], 3: [752, 770, 1051, 1069, 1242, 1340, 1470, 1812, 2015, 2073, 2665, 2907, 3182, 3308, 3312, 3475, 3476, 3526, 3589, 3610, 3842, 4048, 4083, 4354, 4387, 4979, 5074, 5605, 5763, 5795, 5878, 5954]}

glv = GroupLossVariance(X, omega, G2, 1) #axis = 1 (0 rows e 1 columns)
RgrpIU = glv.evaluate(X_est)
print("Group Loss Variance (Rgrp IU):", RgrpIU)