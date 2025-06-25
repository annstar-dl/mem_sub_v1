from utils import read_parameter_from_yaml_file

if __name__ == "__main__":
    r = read_parameter_from_yaml_file("r")
    print(f"Radius: {r}")
    rho = read_parameter_from_yaml_file("rho")
    print(f"Learning rate: {rho}")
    d = read_parameter_from_yaml_file("d")
    print(f"Number of dilation steps: {d}")
    nb_iter = read_parameter_from_yaml_file("nb_iter")
    print(f"Number of iterations: {nb_iter}")
