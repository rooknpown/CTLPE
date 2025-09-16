import math
import torch
import torchcde
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

class NDE(nn.Module):
    def __init__(self):
        super(NDE, self).__init__()
        return

    def forward(self):
        return


class NCDE(NDE):
    def __init__(self, enc_in, enc_out):
        # print("NCDE called")
        super(NCDE, self).__init__()
        # latent_dim = 6
        # units = 100
        self.ncde = NeuralCDE(enc_in, 4, enc_out)
        # self.ncde = NeuralCDE(enc_in, 128, enc_out)
        # ode_func_net = create_net(enc_in, enc_in, 
        #     n_layers = 3, n_units = units, nonlinear = nn.Tanh)
        # gen_ode_func = ODEFunc( 
        # input_dim = enc_in, 
        # latent_dim = latent_dim, 
        # ode_func_net = ode_func_net)
        # self.diffeq_solver = DiffeqSolver(enc_in, gen_ode_func, 'dopri5', latent_dim, 
        # odeint_rtol = 1e-3, odeint_atol = 1e-4)

    def forward(self, timef):
        # print("x shape")
        # print(x.shape) 
        # print(x[0])
        # print(x[0].shape)
        # x = x[x.nonzero().detach()]
        # print("xshape")
        # print(x.shape)
        # print("timef")
        # print(timef)
        out1 = torchcde.hermite_cubic_coefficients_with_backward_differences(timef)
        # print("out1shape")
        # print(out1.shape)
        out2 = self.ncde(out1)
        # print("ncdeout")
        # print(out2)
        # print("out2 shape")
        # print(out2.shape)
        # time_steps_to_predict = torch.tensor(range(100), dtype=float)
        # out3 = self.diffeq_solver(out2, time_steps_to_predict)
        # print("out3 shape")
        # print(out3.shape)
        return out2



class CDEFunc(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)



    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        # print("t shape")
        # print(t.shape)
        # print(t)
        # print("z shape")
        # print(z.shape)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)

        z = z.tanh()
        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        # print("z shape")
        # print(z.shape)
        # print("params")
        # print(z.size(0))
        # print(t.shape[0]-1)

        # z = z.view(z.size(0), t.shape[0]-1, self.hidden_channels, self.input_channels)
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z
    
######################
# Next, we need to package CDEFunc up into a model that computes the integral.
######################
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, interpolation="cubic"):
        super(NeuralCDE, self).__init__()
        # print("input and hidden")
        # print(input_channels)
        # print(hidden_channels)
        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation

    def forward(self, coeffs):
        # print("coeffs")
        # print(coeffs)
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ###################### 
        # print(X.shape)

        X0 = X.evaluate(X.grid_points[0])

        # print("X interval")
        # print(X.interval.shape)
        # print("X's t")
        # print(X.grid_points.shape)
    
        z0 = self.initial(X0)

        ######################
        # Actually solve the CDE.
        ######################
        adjoint_params = tuple(self.func.parameters()) + (coeffs,)

        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=X.grid_points, adjoint_params = adjoint_params)

        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value,
        # and then apply a linear map.
        ######################
        # z_T = z_T[:, 1]
        pred_y = self.readout(z_T)
        # print("predy shape")
        # print(pred_y.shape)
        return pred_y

### code from latent ode
class DiffeqSolver(nn.Module):
	def __init__(self, input_dim, ode_func, method, latents, 
			odeint_rtol = 1e-4, odeint_atol = 1e-5, device = torch.device("cpu")):
		super(DiffeqSolver, self).__init__()

		self.ode_method = method
		self.latents = latents		
		self.device = device
		self.ode_func = ode_func

		self.odeint_rtol = odeint_rtol
		self.odeint_atol = odeint_atol

	def forward(self, first_point, time_steps_to_predict, backwards = False):
		"""
		# Decode the trajectory through ODE Solver
		"""

		pred_y = odeint(self.ode_func, first_point, time_steps_to_predict, 
			rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
		pred_y = pred_y.permute(1,0,2)

		return pred_y


def create_net(n_inputs, n_outputs, n_layers = 1, 
	n_units = 100, nonlinear = nn.Tanh):
	layers = [nn.Linear(n_inputs, n_units)]
	for i in range(n_layers):
		layers.append(nonlinear())
		layers.append(nn.Linear(n_units, n_units))

	layers.append(nonlinear())
	layers.append(nn.Linear(n_units, n_outputs))
	return nn.Sequential(*layers)

class ODEFunc(nn.Module):
	def __init__(self, input_dim, latent_dim, ode_func_net, device = torch.device("cpu")):
		"""
		input_dim: dimensionality of the input
		latent_dim: dimensionality used for ODE. Analog of a continous latent state
		"""
		super(ODEFunc, self).__init__()

		self.input_dim = input_dim
		self.device = device

		init_network_weights(ode_func_net)
		self.gradient_net = ode_func_net

	def forward(self, t_local, y, backwards = False):
		"""
		Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

		t_local: current time point
		y: value at the current time point
		"""
		grad = self.get_ode_gradient_nn(t_local, y)
		if backwards:
			grad = -grad
		return grad

	def get_ode_gradient_nn(self, t_local, y):
		return self.gradient_net(y)

	def sample_next_point_from_prior(self, t_local, y):
		"""
		t_local: current time point
		y: value at the current time point
		"""
		return self.get_ode_gradient_nn(t_local, y)


def init_network_weights(net, std = 0.1):
	for m in net.modules():
		if isinstance(m, nn.Linear):
			nn.init.normal_(m.weight, mean=0, std=std)
			nn.init.constant_(m.bias, val=0)
