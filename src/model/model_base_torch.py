
import torch
from torchdiffeq import odeint
from src.model.eqn_generator import EquationGenerator
from src.model.model_base import EpidemicModelBase
from src.model.model_matrix import MatrixGenerator


def get_n_states(n_classes, comp_name):
    return [f"{comp_name}_{i}" for i in range(n_classes)]


class RostModelHungary(EpidemicModelBase):
    def __init__(self, model_data):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_state_comp = ["l1", "l2",
                         "ip", "ia1", "ia2", "ia3",
                         "is1", "is2", "is3",
                         "ih", "ic", "icr", "c"]
        compartments = ["s"] + self.get_n_compartments(model_data.model_parameters_data) + ["r", "d"]
        super().__init__(model_data=model_data, compartments=compartments)
        self.eq_solver = EquationGenerator(ps=model_data.model_parameters_data,
                                           actual_population=self.population)

    def get_n_compartments(self, params):
        compartments = []
        for comp in self.n_state_comp:
            compartments.append(get_n_states(comp_name=comp, n_classes=params[f'n_{comp}']))
        return [state for n_comp in compartments for state in n_comp]

    def get_model(self,  xs, ts, ps, cm):
        model_eq = self.eq_solver.evaluate_equations(ps, cm=cm, xs=xs)
        return torch.cat(tuple(model_eq))

    def get_n_state_val(self, ps, val):
        n_state_val = dict()
        slice_start = 1
        slice_end = 1
        for comp in self.n_state_comp:
            n_states = ps[f'n_{comp}']
            slice_end += n_states
            n_state_val[comp] = val[slice_start:slice_end]
            slice_start += n_states
        return n_state_val

    def update_initial_values(self, iv, ps):
        iv["l1_2"][2] = 1
        l1_states = get_n_states(n_classes=ps["n_l1"], comp_name="l1")
        l2_states = get_n_states(n_classes=ps["n_l2"], comp_name="l2")

        l1 = torch.stack([iv[state] for state in l1_states]).sum(0)
        l2 = torch.stack([iv[state] for state in l2_states]).sum(0)

        iv.update({
            "s": self.population - (l1 + l2)
        })

    def get_solution_torch(self, t, parameters, cm):
        initial_values = self.get_initial_values(ps=parameters)
        model_wrapper = ModelFun(self, parameters, cm).to(self.device)
        return odeint(model_wrapper.forward, initial_values, t, method='euler')


class ModelFun(torch.nn.Module):
    """
    Wrapper class for VaccinatedModel.get_model. Inherits from torch.nn.Module, enabling
    the use of a GPU for evaluation through the library torchdiffeq.
    """
    def __init__(self, model, ps, cm):
        super(ModelFun, self).__init__()
        self.model = model
        self.ps = ps
        self.cm = cm

    def forward(self, ts, xs):
        return self.model.get_model(ts, xs, self.ps, self.cm)


class RostModelHungary2(EpidemicModelBase):
    def __init__(self, model_data, cm, ps, xs):
        self.ps = ps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.compartments = ["s", "l1", "l2",
                             "ip", "ia1", "ia2", "ia3",
                             "is1", "is2", "is3",
                             "ih", "ic", "icr", "c", "r", "d"]
        super().__init__(model_data=model_data, compartments=self.compartments)
        self.n_comp = len(self.compartments)
        self.matrix_generator = MatrixGenerator(model=self, cm=cm, ps=ps, xs=xs)

    def update_initial_values(self, iv, ps):
        pass

    def get_model(self, xs, ts, ps, cm):
        pass

    def get_solution_torch_test(self, t, param, cm):
        iv = self.initialize()
        initial_values = self.get_initial_values_()
        model_wrapper = ModelEq(self, param, cm).to(self.device)
        return odeint(model_wrapper, initial_values, t, method='euler')

    def get_initial_values_(self):
        size = self.n_age * len(self.compartments)
        iv = torch.zeros(size)
        iv[self.c_idx['l1_2']] = 1
        iv[self.c_idx['s']:size:self.n_comp] = self.population
        iv[0] -= 1
        return iv

    def idx(self, state: str) -> bool:
        return torch.arange(self.n_age * self.n_comp) % self.n_comp == self.c_idx[state]

    def aggregate_by_age_n_state(self, solution, comp):
        result = 0
        for state in get_n_states(self.ps[comp], comp):
            result += solution[-1, self.idx(state)].sum()
        return result

    def aggregate_by_age_(self, solution, comp):
        return solution[-1, self.idx(comp)].sum()


class ModelEq(torch.nn.Module):
    def __init__(self, model: RostModelHungary2, ps: dict, cm: torch.Tensor):
        super(ModelEq, self).__init__()
        self.model = model
        self.cm = cm
        self.ps = ps
        self.device = model.device

        self.matrix_generator = model.matrix_generator
        self.get_matrices()

    # For increased efficiency, we represent the ODE system in the form
    # y' = (A @ y) * (T @ y) + B @ y + (V_1 * y) / (V_2 @ y),
    # saving every tensor in the module state
    def forward(self, t, y: torch.Tensor) -> torch.Tensor:
        return torch.mul(self.A @ y, self.T @ y) + self.B @ y

    def get_matrices(self):
        mtx_gen = self.matrix_generator
        A = mtx_gen.get_A()
        T = mtx_gen.get_T()
        B = mtx_gen.get_B()