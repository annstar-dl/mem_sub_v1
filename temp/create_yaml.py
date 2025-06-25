import os
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap


def create_yaml_file(file_path, data):
    """
    Create a YAML file with the given data.

    Args:
        file_path (str): The path where the YAML file will be created.
        data (dict): The data to be written to the YAML file.
    """

    # Initialize YAML object
    yaml = YAML()
    with open(file_path, 'w') as file:
        yaml.dump(data, file)
    print(f"YAML file created at {file_path}")

if __name__ == "__main__":
    data = CommentedMap()
    data['description'] = 'Parameters of Membrane Subtraction Algorithm'
    data['version'] = 1.0
    data["r"] = 20  # Grid radius, the radius of the neighborhood around the grid point
    data['nb_iter'] = 3  # Number of iterations for fitting the membranes with basis functions
    data['d'] = 4  # Number dilation steps for the mask enlargement
    data['w'] = 4  # Stride for the sampling grid
    data['max_nb_iter_GD'] = 30  # Maximum number of iterations of Gradient descent
    data['rho'] = 0.025  # Learning rate for gradient descent
    data.yaml_add_eol_comment('Grid radius, the radius of the neighborhood around the grid point', key='r')
    data.yaml_add_eol_comment('Number of iterations for fitting the membranes with basis functions', key='nb_iter')
    data.yaml_add_eol_comment('Number of dilation steps for the mask enlargement', key='d')
    data.yaml_add_eol_comment('Stride for the sampling grid', key='w')
    data.yaml_add_eol_comment('Maximum number of iterations of Gradient descent', key='max_nb_iter_GD')
    data.yaml_add_eol_comment('Learning rate for Gradient descent', key='rho')



    # Specify the path where the YAML file will be created
    # find the parent directory of the current script
    maindir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_file_path = os.path.join(maindir,'parameters.yml')

    # Create the YAML file
    create_yaml_file(yaml_file_path, data)