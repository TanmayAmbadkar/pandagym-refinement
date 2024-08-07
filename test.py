import numpy as np

def generate_random_point_inside_sphere(center, radius):
    while True:
        # Generate a random point in a cube of side 2*radius centered at the origin
        # and then shift it to the specified center
        random_point = np.random.uniform(-radius, radius, size=3) + center
        
        # Check if the point is inside the sphere
        distance_to_center = np.linalg.norm(random_point - center)
        if distance_to_center <= radius:
            return random_point

# Sphere parameters
center = np.array([0.38, 0, 0.15])
radius = 0.02

# Generate a single random point inside the sphere
random_point = generate_random_point_inside_sphere(center, radius)

# Display the result
print(random_point)
print(np.sqrt(np.sum(([0.38, 0, 0.15] - random_point)**2)))