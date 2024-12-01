# Initialize Pygame
import math
import time

import pygame as pygame

pygame.init()

# Constants for scaling
FONT_SIZE = 24

# Set up the screen (Resizable)
WIDTH, HEIGHT = 600, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption('Action Dial Visualization')
font = pygame.font.Font(None, FONT_SIZE)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Function to draw the dial with scaled radius and center
def draw_dial(action, radius, center):
    """
    Draws the dial with the current action.
    """
    screen.fill(WHITE)

    # Draw the dial circle
    pygame.draw.circle(screen, BLACK, center, radius, 5)

    # Action to angle mapping (interpolated between 0, 1, and 2)
    action_angle = {
        0: 180,  # Action 0 (left)
        1: 0,    # Action 1 (right)
        2: 90    # Action 2 (center)
    }

    # Interpolated angle for the current action (smooth transition)
    angle = action_angle.get(action, 90)

    # Draw the current action's needle
    end_pos = (center[0] + radius * math.cos(math.radians(angle)),
               center[1] + radius * math.sin(math.radians(angle)))
    pygame.draw.line(screen, RED, center, end_pos, 5)

    # Draw labels for actions (0, 1, 2)
    action_labels = ["0", "1", "2"]
    for i, label in enumerate(action_labels):
        angle = 180 - i * 90  # Evenly space them around the dial
        x = center[0] + (radius + 20) * math.cos(math.radians(angle))
        y = center[1] + (radius + 20) * math.sin(math.radians(angle))
        text = font.render(label, True, BLACK)
        screen.blit(text, (x - text.get_width() // 2, y - text.get_height() // 2))


# Function to display the additional information (time, position, packets)
def display_info(time_sec, position, packets_received):
    """
    Displays the time, position, and packet count information below the dial.
    Text will be stacked vertically.
    """
    # Define vertical positions
    vertical_spacing = 40  # Space between each line of text
    base_y = HEIGHT - 150  # Starting Y position for the text (just below the dial)

    # Time
    time_text = font.render(f"Time: {time_sec}s", True, BLACK)
    screen.blit(time_text, (WIDTH // 4, base_y))

    # Position
    position_text = font.render(f"Position: {position}m", True, BLACK)
    screen.blit(position_text, (WIDTH // 4, base_y + vertical_spacing))

    # Packet count
    packet_text = font.render(f"Packets: {packets_received}", True, BLACK)
    screen.blit(packet_text, (WIDTH // 4, base_y + 2 * vertical_spacing))


# Linear interpolation of data (including actions)
def interpolate_data(input_states, actions, sampling_interval=10, frames_per_data_point=10):
    """
    Interpolates the data for position, time, and actions, while keeping packet counts fixed.
    """
    gw_x = [state[0] for state in input_states]  # Position data (gw_x)
    packet_counts = [state[-1] for state in input_states]  # Packet count data (last element in state)
    timesteps = [i * sampling_interval for i in range(len(gw_x))]  # Time data
    action_data = actions

    # Interpolate position, time, and action linearly
    interpolated_gw_x = []
    interpolated_timesteps = []
    interpolated_packet_counts = []
    interpolated_actions = []

    for i in range(len(gw_x) - 1):
        current_gw_x = gw_x[i]
        next_gw_x = gw_x[i + 1]
        current_time = timesteps[i]
        next_time = timesteps[i + 1]
        current_packet = packet_counts[i]
        current_action = action_data[i]
        next_action = action_data[i + 1]

        # Interpolate between data points
        for j in range(frames_per_data_point):
            ratio = j / frames_per_data_point
            interpolated_gw_x.append(current_gw_x + ratio * (next_gw_x - current_gw_x))
            interpolated_timesteps.append(current_time + ratio * (next_time - current_time))
            interpolated_packet_counts.append(current_packet)  # Keep packet count fixed
            interpolated_actions.append(current_action + ratio * (next_action - current_action))  # Interpolate actions

    # Add the last data point (since the loop skips it)
    interpolated_gw_x.append(gw_x[-1])
    interpolated_timesteps.append(timesteps[-1])
    interpolated_packet_counts.append(packet_counts[-1])
    interpolated_actions.append(action_data[-1])

    return interpolated_timesteps, interpolated_gw_x, interpolated_packet_counts, interpolated_actions


# Main function to run the animation
def run_animation(input_states, actions, sampling_interval=10, frames_per_data_point=10):
    """
    Runs the animation for the dial and updates the state with position, packets, and time.
    """
    # Interpolate data for smoother animation
    timesteps, gw_x, packet_counts, interpolated_actions = interpolate_data(input_states, actions, sampling_interval, frames_per_data_point)

    total_frames = len(timesteps)

    # Scale dial based on window size
    radius = min(WIDTH, HEIGHT) // 4
    center = (WIDTH // 2, HEIGHT // 2)

    running = True
    current_idx = 0
    while running and current_idx < total_frames:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get current data point
        current_action = interpolated_actions[current_idx]
        current_time = timesteps[current_idx]
        current_position = gw_x[current_idx]
        current_packets = packet_counts[current_idx]

        # Draw the dial and information
        draw_dial(current_action, radius, center)
        display_info(current_time, current_position, current_packets)

        # Update display
        pygame.display.update()

        # Interpolate smoothly for time and position
        time.sleep(0.1)  # 10 FPS
        current_idx += 1

    pygame.quit()