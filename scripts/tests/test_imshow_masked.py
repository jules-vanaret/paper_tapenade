import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Create a custom colormap
viridis_white = plt.cm.viridis
viridis_white.set_under("white")  # Set color for values below vmin

# Generate sample data
data = np.random.rand(100, 100)

# Replace all 0 values with NaN to ensure they are not colored
data_with_nan = np.where(data < 0.1, np.nan, data)

# Plot the image with imshow
plt.imshow(
    data_with_nan,
    cmap=viridis_white,
    vmin=np.nanmin(data_with_nan),
    vmax=np.nanmax(data_with_nan),
)

# Add colorbar
plt.colorbar()

plt.show()
