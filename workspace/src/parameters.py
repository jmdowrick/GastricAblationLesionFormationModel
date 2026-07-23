import yaml

def load_parameters(filepath: str) -> dict:
  with open(filepath, 'r') as file:
    params = yaml.safe_load(file)

  therm = params['thermal']

  # calculated parameters
  therm['Wb'] = 0.035e-3 * therm['rho_b'] # perfusion rate
  therm['h_b'] = (therm['u_b'] / therm['u_ref']) ** (1.0 / 1.25) * therm['href']
  therm['h_s'] = (therm['u_s'] / therm['u_ref']) ** (1.0 / 1.25) * therm['href']

  return params
