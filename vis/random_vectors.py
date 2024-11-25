from manim import *
import numpy as np
from torchvision import datasets

class MNISTVectorVisualization(Scene):
    def construct(self):
        # Set white background
        background_color = WHITE
        value_color = BLACK
        highlight_color = BLUE

        # Set camera background color
        self.camera.background_color = background_color
        
        # Change all BLACK fill_color to WHITE and vice versa
        mnist = datasets.MNIST('./data', train=True, download=True)
        
        # Get five different digit indices (e.g., for digits 0, 2, 4, 6, 8)
        digit_indices = [
            [i for i, (_, label) in enumerate(mnist) if label == digit][0]
            for digit in [0, 2, 4, 6, 8]  # Changed to 5 different digits
        ]
        
        # Process each digit
        digit_arrays = []
        for idx in digit_indices:
            digit_image = mnist[idx][0]
            # Use numpy's resize function directly
            downsampled = np.array(digit_image.resize((8, 8)))
            downsampled = downsampled/np.max(downsampled)*255
            digit_arrays.append(downsampled)

        # Create five pixel groups
        pixel_groups = VGroup()
        pixel_size = 0.18 # 0.15
        spacing = 2.8 # 2.2
        
        # Calculate total width needed
        total_width = spacing * 4  # 4 spaces between 5 groups
        start_x = -total_width / 2  # Start from negative half of total width

        for group_idx in range(5):
            pixels = VGroup()

            random_array = np.random.uniform(0, 1, (8, 8))
            
            # Initialize pixels in 2D grid
            for i in range(8):
                for j in range(8):
                    pixel = Square(
                        side_length=pixel_size,
                        fill_opacity=1,
                        fill_color=rgb_to_color([
                            1 - random_array[i, j],
                            1 - random_array[i, j],
                            1 - random_array[i, j]
                        ]),
                        stroke_width=0.3,
                        stroke_color=GRAY_C
                    )
                    # Position relative to group center
                    pixel.move_to([
                        (j - 4) * pixel_size,  # Center the grid by offsetting by 4
                        (4 - i) * pixel_size,  # Center the grid by offsetting by 4
                        0
                    ])
                    pixels.add(pixel)

            # Position this group horizontally
            pixels.center().scale(1.7)
            pixels.shift(RIGHT * (start_x + group_idx * spacing))
            pixel_groups.add(pixels)

        self.add(pixel_groups)
        self.wait(1)

        # 1. Random grids -> Random vectors
        for group_idx in range(5):
            # Calculate total height of vector column (64 elements * size * scale)
            total_height = 64 * pixel_size * 0.6
            start_y = total_height / 2

            vector_positions = [
                [(start_x + group_idx * spacing), start_y - i * pixel_size * 0.6, 0]
                for i in range(64)
            ]
            
            self.play(
                pixel_groups[group_idx].animate.scale(1/2),
                run_time=1
            )
            self.play(
                *[pixel_groups[group_idx][i].animate.move_to(pos).scale(0.6)
                  for i, pos in enumerate(vector_positions)],
                run_time=2
            )

        self.wait(1)

        # 1b. Random vectors -> Larger sample of random vectors (fade opacity)
        background_samples = VGroup()
        n_interpolations = 20  # Number of samples between each main vector
        
        # Create samples between each pair of main vectors
        for i in range(4):  # For the 4 gaps between 5 main vectors
            start_pos = pixel_groups[i].get_center()
            end_pos = pixel_groups[i + 1].get_center()
            
            for j in range(n_interpolations):
                # Calculate interpolated position
                t = (j + 1) / (n_interpolations + 1)  # Fraction between vectors
                interp_x = start_pos[0] + t * (end_pos[0] - start_pos[0])
                
                # Create a column of squares
                sample_column = VGroup()
                random_values = np.random.uniform(0, 1, 64)  # Values between 0 and 1
                
                # Calculate total height of vector column
                total_height = 64 * pixel_size * 0.6
                start_y = total_height / 2
                
                for k in range(64):
                    pixel = Square(
                        side_length=pixel_size,
                        fill_opacity=0,
                        fill_color=highlight_color,
                        stroke_width=0.2,
                        stroke_color=GRAY_C
                    ).scale(0.6)
                    pixel.move_to([
                        interp_x,
                        start_y - k * pixel_size * 0.6,
                        0
                    ])
                    # Store the target opacity as a property
                    pixel.target_opacity = random_values[k]
                    sample_column.add(pixel)
                
                background_samples.add(sample_column)
        
        # Add samples to scene (invisible initially)
        self.add(background_samples)
        
        # Fade in all squares with their individual random opacities
        self.play(
            *[square.animate.set_fill(opacity=square.target_opacity)
              for sample in background_samples 
              for square in sample],
            run_time=2
        )
        
        self.wait(1)

        # 2. Random vectors -> MNIST vectors
        for group_idx in range(5):
            self.play(
                *[pixel_groups[group_idx][i].animate.set_fill(
                    color=rgb_to_color([
                        1 - digit_arrays[group_idx][i//8, i%8]/255,
                        1 - digit_arrays[group_idx][i//8, i%8]/255,
                        1 - digit_arrays[group_idx][i//8, i%8]/255
                    ]),
                    opacity=1.0)  # Explicitly set opacity to 1.0
                  for i in range(64)],
                run_time=2
            )

        self.wait(1)

        # 2b. Remaining random vectors -> MNIST vectors (fade opacity)
        # First, ensure all squares are visible but will transition to MNIST values
        for sample_column in background_samples:
            for square in sample_column:
                square.set_stroke(width=0)  # Remove strokes from interpolated samples

        # Calculate all interpolated values for all gaps at once
        all_interpolated_values = []
        for i in range(4):  # For the 4 gaps between 5 main vectors
            start_digit = digit_arrays[i]
            end_digit = digit_arrays[i + 1]
            
            for j in range(n_interpolations):
                t = (j + 1) / (n_interpolations + 1)
                interp_values = start_digit * (1-t) + end_digit * t
                all_interpolated_values.append(interp_values.flatten()/255)  # Normalize to [0,1]
        
        # Single animation to transition all interpolated vectors simultaneously
        self.play(
            *[background_samples[i][k].animate.set_fill(
                opacity=all_interpolated_values[i][k])  # Keep fully opaque
              for i in range(len(background_samples))
              for k in range(64)],
            run_time=2
        )

        self.wait(3)

        # Fade out background samples before grid transformation
        self.play(
            *[square.animate.set_fill(opacity=0)
              for sample in background_samples
              for square in sample],
            run_time=1
        )

        # 3. MNIST vectors -> MNIST grids
        for group_idx in range(5):
            grid_positions = [
                [(j - 4) * pixel_size + (start_x + group_idx * spacing), (4 - i) * pixel_size, 0]
                for i in range(8)
                for j in range(8)
            ]
            
            # Add opacity setting to ensure squares remain fully opaque
            self.play(
                *[pixel_groups[group_idx][i].animate.move_to(pos).scale(1/0.6)
                  for i, pos in enumerate(grid_positions)],
                run_time=2
            )
            
            # Scale the final grid
            self.play(pixel_groups[group_idx].animate.scale(1.7))

        self.wait(2)

        # 4. 8x8 grids -> 16x16 MNIST images (synchronized)
        transform_animations = []
        full_res_groups = VGroup()  # Initialize the VGroup
        
        for group_idx, idx in enumerate(digit_indices):
            # Get full resolution image and resize to 16x16
            digit_image = mnist[idx][0]
            medium_res = np.array(digit_image.resize((16, 16)))
            medium_res = medium_res/np.max(medium_res)*255
            
            # Create medium-res grid
            pixels = VGroup()
            small_pixel_size = pixel_size
            
            for i in range(16):
                for j in range(16):
                    pixel = Square(
                        side_length=small_pixel_size,
                        stroke_width=0.,
                        fill_opacity=1.0,
                        fill_color=rgb_to_color([
                            1 - medium_res[i, j]/255,
                            1 - medium_res[i, j]/255,
                            1 - medium_res[i, j]/255
                        ])
                    )
                    pixel.move_to([
                        j * pixel.side_length,
                        -i * pixel.side_length,
                        0
                    ])
                    pixels.add(pixel)
            
            target_center = pixel_groups[group_idx].get_center()
            pixels.center()
            pixels.scale(1)
            pixels.move_to(target_center)
            full_res_groups.add(pixels)
            
            # Collect transform animations instead of playing them
            transform_animations.append(FadeTransform(pixel_groups[group_idx], pixels))

        # Play all transformations simultaneously
        self.play(*transform_animations, run_time=2)

        self.wait(2)
