import scipy
import h5py


def load_image_from_mat_h5py(file_path, provided_variable_names='image'):
    """
    Load an image from a .mat file.

    Args:
        file_path (str): Path to the .mat file.
        provided_variable_names (str or list of str): Name(s) of the variable(s) to load from the .mat file.
    Returns:
        np.ndarray: The loaded image as a NumPy array.
    """
    d_vars = {}
    with h5py.File(file_path, 'r') as f:
        var_names =  list(f.keys())
        for var_name in var_names:
            if var_name not in provided_variable_names:
                raise KeyError(f"Variable '{var_name}' not found in the .mat file.")
            else:
                image = f[var_name][:]
                d_vars[var_name] = image
    return d_vars

def load_image_from_mat(file_path, variable_names='image'):
    """
    Load an image from a .mat file.

    Args:
        file_path (str): Path to the .mat file.
    Returns:
        np.ndarray: The loaded image as a NumPy array.
    """
    mat_data = scipy.io.loadmat(file_path)
    if not isinstance(variable_names, list):
        variable_names = [variable_names]
    variables = []
    for variable_name in variable_names:
        if variable_name not in mat_data:
            raise KeyError(f"Variable '{variable_name}' not found in the .mat file.")
        variables.append(mat_data[variable_name])
    return tuple(variables) if len(variables) > 1 else variables[0]