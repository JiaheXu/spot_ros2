from __future__ import print_function
import pinocchio as pin
import hppfcl as fcl

import time
import os
from os.path import dirname, join, abspath

pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")
print("pinocchio_model_dir: ", pinocchio_model_dir)
# model_path = join(pinocchio_model_dir, "example-robot-data/robots")
# model_path = "./spot_description/"
# mesh_dir = "./spot_description/"
# urdf_filename = "./spot_description/spot_arm.urdf"

urdf_model_path =  "./collision/spot_arm.urdf"
srdf_model_path = "./collision/spot_arm.srdf"



# Load model
model = pin.buildModelFromUrdf(urdf_model_path, pin.JointModelFreeFlyer())

q        = pin.randomConfiguration(model)
print('q: %s' % q.T)


# print("data: ", data)



# Load collision geometries
geom_model = pin.buildGeomFromUrdf(
    model, urdf_model_path, pin.GeometryType.COLLISION
)

# Add collisition pairs
geom_model.addAllCollisionPairs()
print("num collision pairs - initial:", len(geom_model.collisionPairs))

# Remove collision pairs listed in the SRDF file



# pin.removeCollisionPairs(model, geom_model, srdf_model_path)
print(
    "num collision pairs - after removing useless collision pairs:",
    len(geom_model.collisionPairs),
)

start = time.time()
# Load reference configuration
# pin.loadReferenceConfigurations(model, srdf_model_path)
pin.loadReferenceConfigurations(model, srdf_model_path)

# Retrieve the half sitting position from the SRDF file
q = model.referenceConfigurations["test"]

q[-7:-1] = 2.3 
print("q_last7: ", q[-7:])

# q = [-1, 1, 0, -1, 1]
# Create data structures
data = model.createData()
geom_data = pin.GeometryData(geom_model)


# Compute all the collisions
pin.computeCollisions(model, data, geom_model, geom_data, q, False)

# Print the status of collision for all collision pairs
for k in range(len(geom_model.collisionPairs)):
    cr = geom_data.collisionResults[k]
    cp = geom_model.collisionPairs[k]
    print(
        "collision pair:",
        cp.first,
        ",",
        cp.second,
        "- collision:",
        "Yes" if cr.isCollision() else "No",
    )

# Compute for a single pair of collision
pin.updateGeometryPlacements(model, data, geom_model, geom_data, q)
pin.computeCollision(geom_model, geom_data, 0)

end = time.time()

print("time cost: ", end - start)