import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Set a random seed for reproducibility
np.random.seed(123)

# Constants for the ocean dimensions, search parameters, and optimization coefficients
OCEAN_WIDTH = 5
OCEAN_LENGTH = 10
NUMBER_OF_SEARCHING_SQUAD = 7
INERTIA_WEIGHT = 0.5
COGNITIVE_COEFFICIENT_C1 = 0.1
SOCIAL_COEFFICIENT_C2 = 0.4
MAXIMUM_SEARCH_ITERATIONS = 50


# Objective function representing the ocean depth
# Create lists to store particle data
particle_data = []


def the_ocean_depth(x, y):
    "Objective function"
    return 60-( (x - 5) ** 2 + (y - 6) ** 2 + np.sin(3 * x + 2) + np.sin(4 * y - 7))


# Initialize particle positions and velocities
def initialize_particles():
    X = np.random.rand(2, NUMBER_OF_SEARCHING_SQUAD)
    X[0] *= OCEAN_WIDTH
    X[1] *= OCEAN_LENGTH
    V = np.random.randn(2, NUMBER_OF_SEARCHING_SQUAD) * 0.1
    print('x')
    print(X)
    print('v')
    print(V)
    return X, V


# Update particle velocities based on cognitive and social influences
def update_velocity(X, V, pbest, gbest):
    r1, r2 = np.random.rand(2)

    particle_momentum_effect = INERTIA_WEIGHT * V
    how_far_from_pbest = pbest - X
    particle_best_position_effect = COGNITIVE_COEFFICIENT_C1 * r1 * how_far_from_pbest
    how_far_from_gbest = gbest.reshape(-1, 1) - X
    global_best_position_effect = SOCIAL_COEFFICIENT_C2 * r2 * how_far_from_gbest

    return (
        particle_momentum_effect
        + particle_best_position_effect
        + global_best_position_effect
    )


# Update particle positions based on velocities
def update_particle_positions(X, V):
    return X + V


# Update personal best positions and fitness values
def update_personal_best(X, pbest, pbest_fitness):
    depths = the_ocean_depth(X[0], X[1])
    update_indices = pbest_fitness <= depths
    pbest[:, update_indices] = X[:, update_indices]
    pbest_fitness = np.maximum(pbest_fitness, depths)
    return pbest, pbest_fitness


# Update global best position and fitness value
def update_global_best(pbest, pbest_fitness):
    best_particle_index = np.argmax(pbest_fitness)
    gbest = pbest[:, best_particle_index]
    gbest_fitness = pbest_fitness[best_particle_index]
    return gbest, gbest_fitness


# Update the search for a single iteration
def update_search_iteration():
    global X, V, pbest, pbest_fitness, gbest, gbest_fitness
    V = update_velocity(X, V, pbest, gbest)
    X = update_particle_positions(X, V)
    pbest, pbest_fitness = update_personal_best(X, pbest, pbest_fitness)
    gbest, gbest_fitness = update_global_best(pbest, pbest_fitness)


# Update scatter plot data
def update_scatter_plot(plot, data, **kwargs):
    if plot is None:
        return ax.scatter(data[0], data[1], **kwargs)
    else:
        plot.set_offsets(data.T)
        return plot


# Update quiver plot data
def update_quiver_plot(arrow, position, velocity, **kwargs):
    if arrow is None:
        return ax.quiver(
            position[0],
            position[1],
            velocity[0],
            velocity[1],
            **kwargs,
        )
    else:
        arrow.set_offsets(position.T)
        arrow.set_UVC(velocity[0], velocity[1])
        return arrow


# Animation function for updating the plot and saving frames
def animate(i):
    global pbest_plot, p_plot, p_arrow, gbest_plot
    "Steps of PSO: algorithm update and show in plot"
    title = f"iteration {i:02d}"

    # Save the current frame as an image
    frame_filename = os.path.join(frames_dir, f"frame_{i:03d}.png")
    fig.savefig(frame_filename, dpi=120)

    update_search_iteration()
    ax.set_title(title,backgroundcolor='white')

    p_plot = update_scatter_plot(p_plot, X, marker="o", color="gray")
    p_arrow = update_quiver_plot(p_arrow, X, V, color="gray", width=0.005)
    pbest_plot = update_scatter_plot(
        pbest_plot, pbest, marker="o", color="yellow"
    )
    gbest_plot = update_scatter_plot(
        gbest_plot, gbest, marker="*", s=100, color="red"
    )

    # Append particle data to the list
    particle_data.append(
        {
            "Iteration": i,
            "ParticlePositions_x": X[0].tolist(),
            "ParticlePositions_y": X[1].tolist(),
            "Velocities_x": V[0].tolist(),
            "Velocities_y": V[1].tolist(),
            "PersonalBests": pbest.tolist(),
            "GlobalBest": gbest.tolist(),
        }
    )

    return ax, pbest_plot, p_plot, p_arrow, gbest_plot


def write_to_csv_file(particle_data, csv_particle_filename):
    with open(csv_particle_filename, "w", newline="") as csvfile:
        fieldnames = [
            "Iteration",
            "ParticlePositions_x",
            "ParticlePositions_y",
            "Velocities_x",
            "Velocities_y",
            "PersonalBests",
            "GlobalBest",
        ]
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(particle_data)

    print("Particle data saved to:", csv_particle_filename)


# Generate a meshgrid for the ocean depths
x_array = np.linspace(0, OCEAN_WIDTH, 100)
y_array = np.linspace(0, OCEAN_LENGTH, 100)
x, y = np.array(np.meshgrid(x_array, y_array))
depth = the_ocean_depth(x, y)

# Find the minimum depth coordinates
x_max, y_max = x.ravel()[depth.argmax()], y.ravel()[depth.argmax()]

# Initialize particle positions and personal/global bests
X, V = initialize_particles()
pbest = X
pbest_fitness = the_ocean_depth(pbest[0], pbest[1])
gbest = pbest[:, np.argmax(pbest_fitness)]
gbest_fitness = np.max(pbest_fitness)
print(gbest)

# Create the figure and plot the ocean map
fig, ax = plt.subplots(figsize=(8, 6))

# Create the animation without saving as GIF
anim = FuncAnimation(
    fig,
    animate,
    frames=list(range(1, MAXIMUM_SEARCH_ITERATIONS + 1)),
    interval=500,
    blit=False,
    repeat=False,
)

# Create a directory to save animation frames
frames_dir = r"animation_frames"
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)


img = ax.imshow(
    depth, extent=[0, OCEAN_WIDTH, 0, OCEAN_LENGTH], origin="lower", cmap="Blues"
)
fig.colorbar(img, ax=ax)
ax.plot([x_max], [y_max], marker="x", markersize=10, color="orange")
contours = ax.contour(x, y, depth, 10, colors="black", alpha=0.4)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")

# Initialize variables for the scatter and quiver plots
pbest_plot = None
p_plot = None
p_arrow = None
gbest_plot = None

# Create the animation
anim = FuncAnimation(
    fig,
    animate,
    frames=list(range(1, MAXIMUM_SEARCH_ITERATIONS + 1)),
    interval=500,
    blit=False,
    repeat=True,
)
anim.save(r"PSO.gif", dpi=300, writer="imagemagick",savefig_kwargs={'transparent': True})

# Print the optimal solutions found by the algorithm
print(f"AI found NEMO at found at x:{round(gbest[0])},  y:{round(gbest[0])}")
print(f"NEMO was at x: {round(x_max,2)}, y:{round(y_max,2)}")
# Write particle data to a CSV file
write_to_csv_file(
    particle_data,
    r"particle_data.csv",
)
