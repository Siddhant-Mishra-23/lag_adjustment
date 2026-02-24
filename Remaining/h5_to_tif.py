import os
import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin


def read_h5_data(h5_path: str):
    with h5py.File(h5_path, 'r') as h5_file:
            sm_surface_analysis = h5_file['Geophysical_Data/sm_surface'][:]
            lat = h5_file['cell_lat'][:]
            lon = h5_file['cell_lon'][:]
            return sm_surface_analysis, lat, lon

def preprocess_data(data: np.ndarray) -> np.ndarray:
    """Masks missing values and normalizes data to 0–255."""
    data = np.where(data == -9999.0, np.nan, data)
    valid_min = np.nanmin(data)
    valid_max = np.nanmax(data)
    norm = (data - valid_min) / (valid_max - valid_min) * 255
    norm = np.clip(norm, 0, 255)
    return np.nan_to_num(norm, nan=0).astype(np.uint8)

def create_geotransform(lat: np.ndarray, lon: np.ndarray, pixel_size: float = 0.081):
    """Creates a geotransform matrix for raster writing."""
    top_left_lon = lon.min()
    top_left_lat = lat.max()
    return from_origin(top_left_lon, top_left_lat, pixel_size, pixel_size)


def write_geotiff(output_path: str, data: np.ndarray, transform, crs: str = "EPSG:4326"):
    """Writes a 2D numpy array to a GeoTIFF file."""
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(data, 1)


def process_smap_file(directory: str, output_dir: str,h5_path: str = None):
    for filename in os.listdir(directory):
        h5_path = os.path.join(directory,filename)
        tiff_output_path = os.path.join(output_dir, filename)
        sm_data, lat, lon = read_h5_data(h5_path)
        processed_data = preprocess_data(sm_data)
        transform = create_geotransform(lat, lon)
        write_geotiff(tiff_output_path, processed_data, transform)
        print(f"GeoTIFF saved with coordinate info: {tiff_output_path}")
        return tiff_output_path


# Example usage
if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    print("Current script directory:", current_path)
    smap_dir = "AHRC-Sata_Download"
    tiff_dir = "AHRC_tiff_files"
    if os.path.exists(smap_dir):
        print("Path exists.") 
    else:
        print("Path does not exist. Creating directory.")
        os.makedirs(smap_dir)
    process_smap_file(directory= smap_dir,output_dir=tiff_dir)
