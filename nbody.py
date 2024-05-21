import numpy as np


def read_parameters(filename):
    """
    Read the parameters file and return it's values as a dict.

    """
    parameters = {
        "nparticle": 2,
        "min_mass": 1.0,
        "max_mass": 1.0,
        "timestep": 1e-4,
        "max_time": 1e-4,
        "epsilon": 1e-5,
        "write_frequency": 1,
        "seed": 1,
        "finite_domain": False,
        "centre_of_mass": True,
        "bound_state": True,
        "xmin": -1.0,
        "xmax": 1.0,
        "ymin": -1.0,
        "ymax": 1.0,
        "zmin": -1.0,
        "zmax": 1.0,
    }
    with open(filename, "r") as file:
        for line in file.readlines():
            line = line.lstrip()
            if line.startswith("#"):
                continue
            key, val = line.split("=")
            key = key.strip()
            if isinstance(parameters[key], bool):
                parameters[key] = val.strip() == "yes"
            elif isinstance(parameters[key], int):
                parameters[key] = int(val)
            elif isinstance(parameters[key], float):
                parameters[key] = float(val)

    assert parameters["max_time"] > 1e-15
    assert parameters["nparticle"] > 1
    assert parameters["max_mass"] > parameters["min_mass"] > 0
    assert parameters["timestep"] > 1e-15
    assert parameters["write_frequency"] > 0
    assert parameters["seed"] > 0
    assert 0.1 > parameters["epsilon"] > 1e-15
    assert parameters["xmax"] > parameters["xmin"]
    assert parameters["ymax"] > parameters["ymin"]
    assert parameters["zmax"] > parameters["zmin"]
    assert not parameters["finite_domain"], "finite_domain not supported"

    return parameters


def write_state(timestep, x):
    filename = f"nbody_{timestep}.mol"
    Np = x.shape[0]
    lines = [f"nbody_{timestep}"]
    lines.append("  MOE2000")
    lines.append("")
    lines.append(f"{Np:3} 0  0  0  0  0  0  0  0  0   1 V2000")
    for i in range(Np):
        lines.append(
            f"{x[i, 0]:10.4f}{x[i, 0]:10.4f}{x[i, 0]:10.4f} C   0  0  0  0  0  0  0  0  0  0  0  0"
        )
    lines.append("M  END")
    lines.append("$$$$")
    with open(filename, "w") as f:
        f.writelines("\n".join(lines))


def print_state(l, x, v, mass, dt):
    U = compute_potential_energy(x, v, mass)
    K = compute_kinetic_energy(x, v, mass)
    total_energy = K - U
    print(l * dt, total_energy / x.shape[0])
    write_state(l, x)


def scale(arr, low, high):
    arr *= high - low
    arr += low


def center_particles(arr, mass):
    center_mass = (mass @ arr) / mass.sum()
    arr -= center_mass


def compute_potential_energy(x, v, mass):
    U = 0.0
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            rij = np.sqrt(((x[i] - x[j]) ** 2).sum() + parameters["epsilon"])
            U += mass[i] * mass[j] / rij
    return U


def compute_kinetic_energy(x, v, mass):
    return 0.5 * np.sum(mass @ (v**2))


def set_initial_value(parameters):
    gen = np.random.default_rng(parameters["seed"])
    Np = parameters["nparticle"]
    X = gen.random((Np, 3))
    scale(X[:, 0], parameters["xmin"], parameters["xmax"])
    scale(X[:, 1], parameters["ymin"], parameters["ymax"])
    scale(X[:, 2], parameters["zmin"], parameters["zmax"])
    dX = gen.random((Np, 3))
    scale(dX, -0.2, 0.2)
    mass = gen.random(Np)
    scale(Mass, parameters["min_mass"], parameters["max_mass"])

    # Add rotation
    dX[:, 1] += X[:, 0] / 10.0
    dX[:, 0] -= X[:, 1] / 10.0

    if parameters["centre_of_mass"]:
        # Set the center of mass and it's speed to 0
        center_particles(X, mass)
        center_particles(dX, mass)

    if parameters["bound_state"]:
        # Make sure that the total energy of the system is negative
        # so particle don't fly in the distance
        U = compute_potential_energy(X, dX, mass)
        K = compute_kinetic_energy(X, dX, mass)
        alpha = np.sqrt(U / (2.0 * K))
        dX *= alpha

    return X, dX, mass


def compute_acceleration(x, mass, epsilon):
    acc = np.zeros_like(x)
    sum = np.empty(3, dtype=np.double)
    for i in range(x.shape[0]):
        sum[:] = 0
        for j in range(x.shape[0]):
            if i == j:
                continue
            rij = np.sqrt(((x[i] - x[j]) ** 2).sum() + epsilon)
            p = mass[j] / rij**3
            sum += p * (x[i] - x[j])
        acc[i] = -sum
    return acc


def step_verlet(x, v, acc, mass, dt, epsilon):
    xnew = x + dt * v + 0.5 * dt * dt * acc
    accnew = compute_acceleration(xnew, mass, epsilon)
    vnew = v + 0.5 * dt * (acc + accnew)
    x[:] = xnew[:]
    v[:] = vnew[:]
    acc[:] = accnew[:]


def integrate(x, v, mass, parameter):
    dt = parameters["timestep"]
    Np = parameters["nparticle"]
    Nt = int(parameters["max_time"] // dt)
    write_frequency = parameters["write_frequency"]
    epsilon = parameters["epsilon"]

    acc = compute_acceleration(x, mass, epsilon)
    for l in range(Nt):
        if l % write_frequency == 0:
            print_state(l, x, v, mass, dt)
        step_verlet(x, v, acc, mass, dt, epsilon)
    write_state(Nt, x)


def main():
    """
    Programme principal
    """

    if len(sys.argv) != 3:
        sys.exit(f"Utilisation: {sys.argv[0]}  parameters.txt")

    filename = sys.argv[1]
    parameters = read_parameters(filename)
    x, v, m = set_initial_value(parameters)
    integrate(x, v, m, parameters)


if __name__ == "__main__":
    main()
