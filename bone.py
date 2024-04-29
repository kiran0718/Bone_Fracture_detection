import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Function to display images
def cv2_imshow(a):
    if a is None:
        print("Error: Image is empty.")
        return
    a = a.clip(0, 255).astype('uint8')
    plt.imshow(a)
    plt.axis('off')
    plt.show()

# Load the image
image_path = '/images.jpeg'
image = cv2.imread(image_path)
if image is None:
    print("Error: Unable to load the image.")
    exit()
else:
    print("Loaded Image:")

# Display the loaded image
cv2_imshow(image)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
if gray is None:
    print("Error: Unable to convert the image to grayscale.")
    exit()

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Perform dilation to close gaps in the edges
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)

# Find contours in the dilated image
contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate histogram of crack widths
crack_widths = []
for contour in contours:
    (x, y), radius = cv2.minEnclosingCircle(contour)
    crack_width = int(2 * radius)
    crack_widths.append(crack_width)

# Draw rectangles around the cracks
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the histogram
plt.hist(crack_widths, bins=20, color='blue', alpha=0.7)
plt.title("Histogram of Crack Widths")
plt.xlabel('Width')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Number of cracks detected
num_cracks = len(contours)
print("Number of Cracks Detected:", num_cracks)

# Determine if plaster is needed
if num_cracks < 10:
    print("Plaster is recommended.")
else:
    print("You may need rod surgery.")

# 3D Reconstruction
# Depth map generation (simple intensity-based method)
depth_map = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

# 3D reconstruction (create a point cloud)
height, width = gray.shape
y_coords, x_coords = np.mgrid[0:height, 0:width]
point_cloud = np.dstack((x_coords, y_coords, depth_map))

# Create mesh grid for surface plot
X, Y = np.meshgrid(range(width), range(height))

# Create surface plot
fig = go.Figure(data=[go.Surface(x=X, y=Y, z=depth_map)])

# Update layout for better visualization
fig.update_layout(
    title="3D Model",
    scene=dict(
        aspectmode='data',
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
    )
)

# Show the 3D model
fig.show()
