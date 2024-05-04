import base64
import csv

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
    image_file (str): The path to the image file to be used as the background.

    Returns:
    None
    """
    # Open the image file and read the data
    with open(image_file, "rb") as f:
        img_data = f.read()

    # Encode the image data using base64
    b64_encoded = base64.b64encode(img_data).decode()

    # Create a CSS style string to set the background image
    style = f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{b64_encoded});
        background-size: cover;
    }}
    </style>
    """

    # Use Streamlit's markdown function to apply the CSS style
    from streamlit import markdown
    markdown(style, unsafe_allow_html=True)

def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
    results (dict): Dictionary containing the results.
    output_path (str): Path to the output CSV file.
    """
    # Open the output CSV file for writing
    with open(output_path, 'w', newline='') as f:
        # Define the fieldnames for the CSV file
        fieldnames = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number']

        # Create a CSV writer object
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Iterate over the results dictionary and write each row to the CSV file
        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    writer.writerow({
                        'frame_nmr': frame_nmr,
                        'car_id': car_id,
                        'car_bbox': results[frame_nmr][car_id]['car']['bbox'],
                        'license_plate_bbox': results[frame_nmr][car_id]['license_plate']['bbox'],
                        'license_plate_bbox_score': results[frame_nmr][car_id]['license_plate']['bbox_score'],
                        'license_number': results[frame_nmr][car_id]['license_plate']['text']
                    })