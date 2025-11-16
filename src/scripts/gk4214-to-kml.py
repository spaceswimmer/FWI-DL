import numpy as np
from pyproj import Transformer
from scipy.spatial import ConvexHull
import simplekml
import argparse
import os

def read_coordinates(filename):
    """
    Read coordinates from file. Assumes format: x y (space separated)
    """
    points = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        points.append([x, y])
                    except ValueError:
                        continue
    return np.array(points)

def remove_duplicate_points(points, tolerance=0.01):
    """
    Remove duplicate or very close points
    """
    if len(points) == 0:
        return points
    
    # Round points to tolerance to identify duplicates
    rounded_points = np.round(points / tolerance) * tolerance
    _, unique_indices = np.unique(rounded_points, axis=0, return_index=True)
    return points[unique_indices]

def create_outline_polygon(points):
    """
    Create convex hull outline from points with error handling
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to create a polygon")
    
    # Remove duplicate points
    points = remove_duplicate_points(points)
    
    if len(points) < 3:
        raise ValueError("Need at least 3 unique points to create a polygon")
    
    try:
        # Try creating convex hull with standard method
        hull = ConvexHull(points)
        outline_points = points[hull.vertices]
    except Exception as e:
        print(f"ConvexHull failed: {e}. Trying alternative methods...")
        
        # Method 1: Try with QHull options for handling precision issues
        try:
            hull = ConvexHull(points, qhull_options='QbB QJ Qc')
            outline_points = points[hull.vertices]
        except Exception as e2:
            print(f"QHull with options failed: {e2}. Using bounding box...")
            
            # Method 2: Fall back to bounding box
            outline_points = create_bounding_box(points)
    
    # Close the polygon by adding the first point at the end
    outline_points = np.vstack([outline_points, outline_points[0]])
    
    return outline_points

def create_bounding_box(points):
    """
    Create a simple bounding box as fallback
    """
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    
    return np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ])

def convert_coordinates(points, source_crs, target_crs="EPSG:4326"):
    """
    Convert coordinates between different CRS
    """
    try:
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    except Exception as e:
        raise ValueError(f"Could not initialize coordinate transformer from {source_crs} to {target_crs}: {e}")
    
    converted_points = []
    for point in points:
        try:
            lon, lat = transformer.transform(point[0], point[1])
            converted_points.append([lon, lat, 0])  # Add altitude 0
        except Exception as e:
            print(f"Warning: Could not transform point {point}: {e}")
            continue
    
    return np.array(converted_points)

def get_crs_from_args(args):
    """
    Determine the source CRS based on command line arguments
    """
    if args.no_conversion:
        return None  # No conversion needed
    
    if args.utm_zone:
        if args.utm_hemisphere.lower() == 's':
            return f"EPSG:327{args.utm_zone:02d}"  # UTM Southern Hemisphere
        else:
            return f"EPSG:326{args.utm_zone:02d}"  # UTM Northern Hemisphere
    
    # Default to GK Zone 4
    return "EPSG:31468"

def create_kml_file(points, output_filename, original_points_count, is_latlon=True):
    """
    Create KML file with polygon
    """
    kml = simplekml.Kml()
    
    # Create polygon
    polygon_name = f"Points Outline ({original_points_count} points)"
    polygon = kml.newpolygon(name=polygon_name)
    
    if is_latlon:
        # Points are already in WGS84 (lat/lon)
        coords = [(point[0], point[1], point[2]) for point in points]
    else:
        # Points are in projected CRS, need to be converted for description
        coords = [(point[0], point[1], point[2]) for point in points]
    
    polygon.outerboundaryis = coords
    
    # Style the polygon
    polygon.style.polystyle.color = simplekml.Color.changealphaint(100, simplekml.Color.blue)
    polygon.style.linestyle.color = simplekml.Color.red
    polygon.style.linestyle.width = 3
    
    # Add description
    crs_info = "WGS84 (lat/lon)" if is_latlon else "Original Projected CRS"
    polygon.description = f"Outline polygon created from {original_points_count} input points\nCRS: {crs_info}\nTotal outline points: {len(points)}"
    
    kml.save(output_filename)
    print(f"KML file saved as: {output_filename}")

def main():
    parser = argparse.ArgumentParser(
        description='Create outline polygon from coordinates and convert to KML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Coordinate System Options:
  --no-conversion    Use original coordinates without conversion (for projected CRS)
  --utm-zone ZONE    Specify UTM zone (e.g., 44 for UTM Zone 44)
  --utm-hemisphere   Specify hemisphere (n/north or s/south, default: n)
  
If no options are specified, defaults to GK Zone 4 (EPSG:31468)

Examples:
  python script.py points.txt                    # GK Zone 4
  python script.py points.txt --utm-zone 44     # UTM Zone 44N
  python script.py points.txt --no-conversion   # Use original coordinates
        """
    )
    
    parser.add_argument('input_file', help='Input file with coordinates (x y format)')
    parser.add_argument('-o', '--output', default='output_polygon.kml', help='Output KML filename')
    
    # Coordinate system options
    parser.add_argument('--no-conversion', action='store_true', 
                       help='Use original coordinates without conversion to WGS84')
    parser.add_argument('--utm-zone', type=int, 
                       help='UTM zone number (e.g., 44 for UTM Zone 44N)')
    parser.add_argument('--utm-hemisphere', choices=['n', 's', 'north', 'south'], 
                       default='n', help='UTM hemisphere (default: n)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return
    
    try:
        # Read coordinates
        print("Reading coordinates...")
        points = read_coordinates(args.input_file)
        original_count = len(points)
        print(f"Read {original_count} points")
        
        if original_count < 3:
            print("Error: Need at least 3 points to create a polygon")
            return
        
        # Remove duplicates and check again
        points = remove_duplicate_points(points)
        unique_count = len(points)
        print(f"After removing duplicates: {unique_count} unique points")
        
        if unique_count < 3:
            print("Error: Need at least 3 unique points to create a polygon")
            return
        
        # Create outline polygon
        print("Creating outline polygon...")
        outline_points = create_outline_polygon(points)
        print(f"Outline polygon has {len(outline_points)} points")
        
        # Determine CRS and convert if needed
        source_crs = get_crs_from_args(args)
        
        if source_crs is None:
            # No conversion - use original coordinates
            print("Using original coordinates without conversion")
            final_points = outline_points
            # Add zero altitude
            final_points = np.column_stack([final_points, np.zeros(len(final_points))])
            is_latlon = False
        else:
            # Convert to WGS84
            print(f"Converting from {source_crs} to WGS84...")
            final_points = convert_coordinates(outline_points, source_crs)
            is_latlon = True
        
        if len(final_points) == 0:
            print("Error: No points were successfully processed")
            return
        
        # Create KML file
        print("Creating KML file...")
        create_kml_file(final_points, args.output, original_count, is_latlon)
        
        print("Done!")
        print(f"Summary:")
        print(f"- Original points: {original_count}")
        print(f"- Unique points: {unique_count}")
        print(f"- Outline points: {len(final_points)}")
        if source_crs:
            print(f"- Coordinate system: {source_crs} → WGS84")
        else:
            print(f"- Coordinate system: Original (no conversion)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()