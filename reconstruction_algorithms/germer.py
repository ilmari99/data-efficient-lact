import numpy as np
import torch
import kornia.geometry.transform
from germer_model import Model

def reconstruct(sinogram, angles):
    """
    Reconstructs an image from a given sinogram using a pretrained neural network.
    
    The sinogram is expected to be a NumPy array of shape (num_angles, 560), and 
    angles a 1D NumPy array of acquisition angles (in degrees). The function:
      1. Loads the network model from "model.pth".
      2. Formats the sinogram into a two-channel input where the first channel contains
         the sinogram (padded to 181 angles if needed) and the second channel marks valid
         sinogram rows.
      3. Runs the model in evaluation mode and rotates the predicted image using the first angle.
      4. Thresholds the output (using the mean of the prediction) to produce a binary mask.
    
    Args:
        sinogram (numpy.ndarray): Array of shape (num_angles, 560) containing the sinogram.
        angles (numpy.ndarray): 1D array of angles (in degrees). Its length should match the 
                                number of rows in the sinogram.
    
    Returns:
        numpy.ndarray: A binary (0/1) image representing the reconstruction.
    """
    sinogram = sinogram.cpu().numpy()
    sinogram = sinogram / 255.0
    # Set up device and load the pretrained model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./reconstruction_algorithms/model.pth"
    model = Model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Ensure the sinogram is a NumPy array.
    sinogram = np.asarray(sinogram)
    num_angles = sinogram.shape[0]
    detector_size = sinogram.shape[1]  # Expected to be 560.
    
    # Prepare input: create a (1,2,181,detector_size) array.
    # The first channel contains the sinogram (inserted into the top num_angles rows),
    # and the second channel is a binary mask marking valid sinogram rows.
    input_array = np.zeros((1, 2, 181, detector_size), dtype=np.float32)
    print(num_angles)
    print(f"sinogram shape: {sinogram.shape}")
    input_array[0, 0, :num_angles, :] = sinogram
    input_array[0, 1, :num_angles, :] = 1.0
    
    # Convert to a PyTorch tensor and move to device.
    inputs = torch.tensor(input_array, device=device)
    
    with torch.no_grad():
        predictions = model(inputs)
        # Rotate the prediction based on the first angle.
        # (kornia.geometry.transform.rotate expects the angle in degrees as a tensor.)
        start_angle = torch.tensor(angles[:1], device=device).float()
        predictions = kornia.geometry.transform.rotate(predictions, start_angle)
        # Extract the single reconstructed image from the batch.
        prediction = predictions[0, 0]
        prediction_np = prediction.cpu().numpy()
    
    # Binarize the output: if the image is not already binary, threshold using its mean.
    binary_prediction = (prediction_np > prediction_np.mean()).astype(np.uint8)
    
    return binary_prediction