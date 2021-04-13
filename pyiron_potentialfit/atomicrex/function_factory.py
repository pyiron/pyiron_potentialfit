import xml.etree.ElementTree as ET
import copy

import numpy as np
import matplotlib.pyplot as plt

from pyiron_base import PyironFactory, DataContainer



class FunctionFactory(PyironFactory):
    """
    Class to conveniently create different function objects.
    for detailed information about the function visit the
    atomicrex documentation.
    """    
    @staticmethod
    def user_function(identifier, input_variable="r", species=["*", "*"], is_screening_function=False):
        return UserFunction(identifier, input_variable=input_variable, species=species, is_screening_function=is_screening_function)

    @staticmethod
    def poly(identifier, cutoff, species=["*", "*"]):
        return Poly(identifier, cutoff=cutoff, species=species)

    @staticmethod
    def spline(identifier, cutoff, derivative_left=0, derivative_right=0, species=["*", "*"]):
        return Spline(identifier, cutoff, derivative_left, derivative_right, species)

    @staticmethod
    def exp_A_screening(identifier, cutoff, species=["*", "*"], is_screening_function=True):
        return ExpA(identifier, cutoff, species=species, is_screening_function=is_screening_function)

    @staticmethod
    def exp_B_screening(identifier, cutoff, rc, alpha, exponent, species=["*", "*"], is_screening_function=True):
        return ExpB(identifier, cutoff, rc, alpha, exponent, species=species, is_screening_function=is_screening_function)

    @staticmethod
    def exp_gaussian_screening(identifier, cutoff, stddev, alpha, exponent, species=["*", "*"], is_screening_function=True):
        return ExpGaussian(identifier, cutoff, stddev, alpha, exponent, species=species, is_screening_function=is_screening_function)

    @staticmethod
    def morse_A(identifier, D0, r0, alpha, species=["*","*"]):
        return MorseA(identifier, D0, r0, alpha, species=species)

    @staticmethod
    def morse_B(identifier, D0, r0, beta, S, delta, species=["*","*"]):
        return MorseB(identifier, D0, r0, beta, S, delta, species=species)

    @staticmethod
    def morse_C(identifier, A, B, mu, lambda_val, delta, species=["*", "*"]):
        return MorseC(identifier, A, B, mu, lambda_val, delta, species=species)

    @staticmethod
    def gaussian(identifier, prefactor, eta, mu, species=["*", "*"]):
        return GaussianFunc(identifier, prefactor, eta, mu, species)


class SpecialFunction(DataContainer):
    """
    Analytic functions defined within atomicrex should inherit from this class
    https://atomicrex.org/potentials/functions.html#index-1
    https://atomicrex.org/potentials/functions.html#specialized-functions
    """    
    def __init__(self, identifier=None, species=["*", "*"], is_screening_function=False):
        super().__init__(table_name=f"special_function_{identifier}")
        self.species = species
        self.parameters = FunctionParameterList()
        self.is_screening_function = is_screening_function
        self.identifier = identifier
        if not is_screening_function:
            self.screening = None
            
    def _to_xml_element(self, name):
        if self.is_screening_function:
            screening = ET.Element("screening")
            root = ET.SubElement(screening, f"{name}")
        else:
            root = ET.Element(f"{name}")

        root.set("id", f"{self.identifier}")
        for param in self.parameters.values():
            p = ET.SubElement(root, f"{param.param}")
            p.text = f"{param.start_val}"
        
        # This if condition is to prevent an error with the expA screening function
        # It is a bit hacky and if another function with only 1 parameter is added
        # it probably has to be rewritten 
        if len(self.parameters.values()) > 1:
            root.append(self.parameters.fit_dofs_to_xml_element())

        if not self.is_screening_function:
            if self.screening is not None:
                root.append(self.screening._to_xml_element())
            return root
        else:
            return screening
    
    @property
    def func(self):
        return None

    def plot(self):
        if self.func is None:
            raise NotImplementedError("A func property needs to be defined in the subclass")
        else:
            return plot(self.func)


class Poly(DataContainer):
    """
    Polynomial interpolation function.
    """    
    def __init__(self, identifier=None, cutoff=None, species=["*", "*"]):
        super().__init__(table_name=f"Poly_{identifier}")
        self.identifier = identifier
        self.cutoff = cutoff
        self.species = species
        self.parameters = PolyCoeffList()

    def _to_xml_element(self):
        poly = ET.Element("poly")
        cutoff = ET.SubElement(poly, "cutoff")
        cutoff.text = f"{self.cutoff}"
        poly.append(self.parameters._to_xml_element())
        return poly


class Spline(DataContainer):
    """
    Spline interpolation function
    """
    def __init__(
        self,
        identifier=None,
        cutoff=None,
        derivative_left=0,
        derivative_right=0,
        species = ["*", "*"]
        ):
        super().__init__(table_name=f"Spline_{identifier}")
        self.identifier = identifier
        self.cutoff = cutoff
        self.derivative_left = derivative_left
        self.derivative_right = derivative_right
        self.species = species
        self.parameters = NodeList()

    def _to_xml_element(self):
        spline = ET.Element("spline")
        spline.set("id", self.identifier)
        if self.cutoff is not None:
            cutoff = ET.SubElement(spline, "cutoff")
            cutoff.text = f"{self.cutoff}"
        der_l = ET.SubElement(spline, "derivative-left")
        der_l.text = f"{self.derivative_left}"
        der_r = ET.SubElement(spline, "derivative-right")
        der_r.text = f"{self.derivative_right}"
        spline.append(self.parameters._to_xml_element())
        return spline


class ExpA(SpecialFunction):

    def __init__(self, identifier=None, cutoff=None, species=["*", "*"], is_screening_function=True):
        super().__init__(identifier, species=species, is_screening_function=is_screening_function)
        self.parameters.add_parameter(
            "cutoff",
            start_val=cutoff,
            enabled=False,
            fitable=False,
        )
    
    @property
    def func(self):
        return lambda r: np.exp(1/(r-self.parameters.cutoff.start_val))

    def _to_xml_element(self):
        return super()._to_xml_element(name="exp-A")


class ExpB(SpecialFunction):
    def __init__(
        self,
        identifier = None,
        cutoff = None,
        rc = None,
        alpha = None,
        exponent = None,
        species = None,
        is_screening_function = True
        ):
        super().__init__(identifier, species=species, is_screening_function=is_screening_function)
        self.parameters.add_parameter(
            "cutoff",
            start_val=cutoff,
            enabled=False,
            fitable=False,
        )
        self.parameters.add_parameter(
            "rc",
            start_val=rc,
            enabled=False,
        )
        self.parameters.add_parameter(
            "alpha",
            start_val=alpha,
            enabled=False,
        )
        self.parameters.add_parameter(
            "exponent",
            start_val=exponent,
            enabled=False,
        )

    @property
    def func(self):
        return lambda r: np.exp(-np.sign(
            self.parameters.exponent.start_val) *
            self.parameters.alpha.start_val /
            (1-((r-self.parameters.rc.start_val) /
            self.parameters.cutoff.start_val - self.parameters.rc.start_val) **
            self.parameters.exponent.start_val ))

    def _to_xml_element(self):
        return super()._to_xml_element(name="exp-B")


class ExpGaussian(SpecialFunction):
    def __init__(
        self,
        identifier=None,
        cutoff=None,
        stddev=None,
        alpha=None,
        exponent=None,
        species=["*", "*"],
        is_screening_function=True
        ):
        super().__init__(identifier, species=species, is_screening_function=is_screening_function)
        self.parameters.add_parameter(
            "cutoff",
            start_val=cutoff,
            enabled=False,
            fitable=False,
        )
        self.parameters.add_parameter(
            "stddev",
            start_val=stddev,
            enabled=False,
        )
        self.parameters.add_parameter(
            "alpha",
            start_val=alpha,
            enabled=False,
        )
        self.parameters.add_parameter(
            "exponent",
            start_val=exponent,
            enabled=False,
        )
    
    @property
    def func(self):
        cutoff = self.parameters["cutoff"].start_val
        stddev = self.parameters["stddev"].start_val
        alpha = self.parameters["alpha"].start_val
        exponent = self.parameters["exponent"].start_val
        return lambda r: np.exp(-np.sign(exponent)*
        alpha/(1-(r/cutoff)**exponent)) * np.exp(-r**2/(2*stddev**2))/(stddev*np.sqrt(2*np.pi))


    def _to_xml_element(self):
        return super()._to_xml_element(name="exp-gaussian")


class MorseA(SpecialFunction):
    def __init__(self, identifier=None, D0=None, r0=None, alpha=None, species=["*", "*"]):
        super().__init__(identifier, species=species, is_screening_function=False)
        self.parameters.add_parameter(
            "D0",
            start_val=D0,
            enabled=True,
        )
        self.parameters.add_parameter(
            "r0",
            start_val=r0,
            enabled=True,
        )
        self.parameters.add_parameter(
            "alpha",
            start_val=alpha,
            enabled=True,
        )

    @property
    def func(self):
        return lambda r: self.parameters.D0.start_val*(
            np.exp(-2*self.parameters.alpha.start_val*(r-self.parameters.r0.start_val)) -
            2*np.exp(-self.parameters.alpha.start_val*(r-self.parameters.r0.start_val))
        )

    def _to_xml_element(self):
        return super()._to_xml_element(name="morse-A")


class MorseB(SpecialFunction):
    def __init__(self, identifier=None, D0=None, r0=None, beta=None, S=None, delta=None, species=["*", "*"]):
        super().__init__(identifier, species=species, is_screening_function=False)
        self.parameters.add_parameter(
            "D0",
            start_val=D0,
            enabled=True,
        )
        self.parameters.add_parameter(
            "r0",
            start_val=r0,
            enabled=True,
        )
        self.parameters.add_parameter(
            "beta",
            start_val=beta,
            enabled=True,
        )
        self.parameters.add_parameter(
            "S",
            start_val=S,
            enabled=True,
        )
        self.parameters.add_parameter(
            "delta",
            start_val=delta,
            enabled=True,
        )

    @property
    def func(self):
        D0 = self.parameters.D0.start_val
        r0 = self.parameters.r0.start_val
        S = self.parameters.S.start_val
        beta = self.parameters.beta.start_val
        delta = self.parameters.delta.start_val
        return lambda r: (D0/(S-1) * np.exp(-beta*np.sqrt(2*S)*(r-r0)) -
            D0*S/(S-1)*np.exp(-beta*np.sqrt(2/S)*(r-r0)) + delta)

    def _to_xml_element(self):
        return super()._to_xml_element(name="morse-B")


class MorseC(SpecialFunction):
    def __init__(self, identifier=None, A=None, B=None, mu=None, lambda_val=None, delta=None, species=["*", "*"]):
        super().__init__(identifier, species=species, is_screening_function=False)
        self.parameters.add_parameter(
            "A",
            start_val=A,
            enabled=True,
        )

        self.parameters.add_parameter(
            "B",
            start_val=B,
            enabled=True,
        )

        self.parameters.add_parameter(
            "mu",
            start_val=mu,
            enabled=True,
        )

        self.parameters.add_parameter(
            "lambda",
            start_val=lambda_val,
            enabled=True,
        )

        self.parameters.add_parameter(
            "delta",
            start_val=delta,
            enabled=True,
        )

    @property
    def func(self):
        A = self.parameters["A"].start_val
        B = self.parameters["B"].start_val
        mu = self.parameters["mu"].start_val
        param_lambda = self.parameters["lambda"].start_val
        delta = self.parameters["delta"].start_val
        return lambda r: A*np.exp(-param_lambda*r)-B*np.exp(-mu*r)+delta

    def _to_xml_element(self):
        return super()._to_xml_element(name="morse-C")


## Renamed GaussianFunc to not mess with the gaussian code when loading from hdf5
class GaussianFunc(SpecialFunction):
    def __init__(self, identifier=None, prefactor=None, eta=None, mu=None, species=None):
        super().__init__(identifier, species=species, is_screening_function=False)
        self.parameters.add_parameter(
            "prefactor",
            start_val=prefactor,
            enabled=True,
        )

        self.parameters.add_parameter(
            "eta",
            start_val=eta,
            enabled=True,
        )

        self.parameters.add_parameter(
            "mu",
            start_val=mu,
            enabled=True,
        )

    @property
    def func(self):
        prefactor = self.parameters["prefactor"].start_val
        eta = self.parameters["eta"].start_val
        mu = self.parameters["mu"].start_val
        return lambda r: prefactor*np.exp(-eta*(r-mu)**2)

    def _to_xml_element(self):
        return super()._to_xml_element(name="gaussian")


class UserFunction(DataContainer):
    """
    Analytic functions that are not implemented in atomicrex
    can be provided as user functions.
    All parameters defined in the function should be added using the
    UserFunction.parameters.add_parameter() method.
    """    
    def __init__(self, identifier=None, input_variable=None, species=["*", "*"], is_screening_function=False):
        super().__init__(table_name=f"user_func_{identifier}")
        self.input_variable = input_variable
        self.identifier = identifier
        self.species = species
        self.parameters = FunctionParameterList()
        self.expression = None
        self.derivative = None
        self.is_screening_function = is_screening_function
        if not is_screening_function:
            self.screening = None

    def _to_xml_element(self):
        if self.is_screening_function:
            screening = ET.Element("screening")
            root = ET.SubElement(screening, "user-function")
        else:
            root = ET.Element("user-function")
        root.set("id", f"{self.identifier}")
        input_var = ET.SubElement(root, "input-var")
        input_var.text = f"{self.input_variable}"
        expression = ET.SubElement(root, "expression")
        expression.text = f"{self.expression}"
        derivative = ET.SubElement(root, "derivative")
        derivative.text = f"{self.derivative}"

        for param in self.parameters.values():
            p = ET.SubElement(root, "param")
            p.set("name", f"{param.param}")
            p.text = f"{param.start_val:.6g}"
        
        root.append(self.parameters.fit_dofs_to_xml_element())

        if not self.is_screening_function:
            if self.screening is not None:
                root.append(self.screening._to_xml_element())
            return root
        else:
            return screening


class FunctionParameter(DataContainer):
    """
    Function parameter. For detailed information
    about the attributes see the atomicrex documentation.
    Objects should only be created using the add_parameter method
    of the FunctionParameterList class.
    """    
    def __init__(
        self,
        param=None,
        start_val=None,
        enabled=True,
        reset=False,
        min_val=None,
        max_val=None,
        fitable=True,
        tag=None):
        self.param = param
        self.start_val = start_val
        self.enabled = enabled
        self.reset = reset
        self.min_val = min_val
        self.max_val = max_val
        self.tag = tag
        self.fitable = fitable
        self.final_value = None

    def _to_xml_element(self):
        root = ET.Element(f"{self.param}")
        root.set("enabled", f"{self.enabled}".lower())
        root.set("reset", f"{self.reset}".lower())
        if self.min_val is not None:
            root.set("min", f"{self.min_val:.6g}")
        if self.max_val is not None:
            root.set("max", f"{self.max_val:.6g}")
        if self.tag is not None:
            root.set("tag", f"{self.tag}")
        return root

    def copy_final_to_start_value(self):
        """
        Copies the final value to start_val.

        Raises:
            ValueError: Raises if fitting of the parameter is enabled,
                        but the final value is None. This should only be the case
                        if the job aborted or was not run yet.
        """        
        if self.enabled:
            if self.final_value is None:
                raise ValueError(f"Fitting is enabled for {self.param}, but final value is None.")
            else:
                self.start_val = copy.copy(self.final_value)
            

class FunctionParameterList(DataContainer):
    def __init__(self):
        super().__init__(table_name="FunctionParameterList")

    def add_parameter(self, param, start_val, enabled=True, reset=False, min_val=None, max_val=None, tag=None, fitable=True):
        """
        Add a function parameter named param to a function.
        This needs to be done manually for user functions and
        not for special functions.

        Args:
            param (str): Name of the parameter. Must exactly match the name in the function expression.
            start_val (float): Starting value of the parameter
            enabled (bool, optional): Determines if the paremeter is varied during fitting. Defaults to True.
            reset (bool, optional): Determine if the parameter should be reset every iteration
            Can help with global optimization. Defaults to False.
            min_val (float, optional): Highly recommended for global optimization. Defaults to None.
            max_val (float, optional): Highly recommended for global optimization. Defaults to None.
            tag (str, optional): [description]. Only necessary for ABOP potentials .Defaults to None.
            fitable (bool, optional): [description]. Changing could cause bugs. Defaults to True.
        """        
        self[param] = FunctionParameter(
            param,
            start_val,
            enabled=enabled,
            reset=reset,
            min_val=min_val,
            max_val=max_val,
            tag=tag,
            fitable=fitable,
        )
        
    def fit_dofs_to_xml_element(self):
        """Internal function
        Returns fit dofs as atomicrex xml element.
        """     
        fit_dof = ET.Element("fit-dof")
        for param in self.values():
            if param.fitable:
                fit_dof.append(param._to_xml_element())
        return fit_dof


class PolyCoeff(FunctionParameter):
    """
    Function parameter, but for polynomial interpolation.
    """    
    def __init__(self, n=None, start_val=None, enabled=True, reset=False, min_val=None, max_val=None):
        super().__init__(
            param="coeff",
            start_val=start_val,
            enabled=enabled,
            reset=reset,
            min_val=min_val,
            max_val=max_val,
            fitable=True,
            tag=None)
        self.n = n

    def _to_xml_element(self):
        root = super()._to_xml_element()
        root.set("n", self.n)


class PolyCoeffList(DataContainer):
    def __init__(self):
        super().__init__(table_name="PolyCoeffList")

    def add_coeff(self, n, start_val, enabled=True, reset=False, min_val=None, max_val=None):
        """
        Add a term in the form of a*x^n.

        Args:
            n (int): Order n of the coefficient
            start_val (float): Starting value of a. 
            enabled (bool, optional): Determines if it should be fitted. Defaults to True.
            reset (bool, optional): Determines if it should be reset after each iteration. Defaults to False.
            min_val (float, optional): Highly recommended for global optimization. Defaults to None.
            max_val (float, optional): Highly recommended for global optimization. Defaults to None.
        """        
        self[f"coeff_{n}"] = PolyCoeff(
            n,
            start_val,
            enabled,
            reset,
            min_val,
            max_val,
        )

    def _to_xml_element(self):
        coefficients = ET.Element("coefficients")
        for coeff in self.values():
            coefficients.append(coeff._to_xml_element())
        return coefficients


class Node(FunctionParameter):
    """
    Function parameter, but for spline interpolation.
    """
    def __init__(self, x=None, start_val=None, enabled=True, reset=False, min_val=None, max_val=None):
        super().__init__(
            param="node",
            start_val=start_val,
            enabled=enabled,
            reset=reset,
            min_val=min_val,
            max_val=max_val,
            fitable=True,
            tag=None)
        self.x = x

    def _to_xml_element(self):
        node = super()._to_xml_element()
        node.set("x", f"{self.x:.6g}")
        node.set("y", f"{self.start_val:.6g}")
        return node


class NodeList(DataContainer):
    def __init__(self):
        super().__init__(table_name="NodeList")

    def add_node(self, x, start_val, enabled=True, reset=False, min_val=None, max_val=None):
        """
        Add a node to the spline interpolation function.

        Args:
            x (float): x coordinate of the node. Does not change during fitting.
            start_val (float): Initial y coordinate of the node.
            enabled (bool, optional): Determines if y is changed during fitting. Defaults to True.
            reset (bool, optional): Determines if y should be reset every iteration. Defaults to False.
            min_val (float, optional): Highly recommended for global optimization. Defaults to None.
            max_val (float, optional): Highly recommended for global optimization. Defaults to None.
        """        
        x = float(x)
        # atomicrex rounds output to 6 digits, so this is done here to prevent issues when reading the output.
        x = round(x, 6)
        key = f"node_{x}"
        self[key] = Node(
            x=x,
            start_val=start_val,
            enabled=enabled,
            reset=reset,
            min_val=min_val,
            max_val=max_val,
        )
        return self[key]

    def _to_xml_element(self):
        nodes = ET.Element("nodes")
        for node in self.values():
            nodes.append(node._to_xml_element())
        return nodes

    def create_from_arrays(self, x, y, min_vals=None, max_vals=None):
        """
        Convenience function to create nodes from lists or arrays of values.
        Allows to easily start the fitting process with physically motivated values
        or values taken from previous potentials.
        Creates len(x) nodes at position x with starting values y.
        All given arrays must have the same length.

        Args:
            x (list or array): x values of the nodes
            y (list or array): corresponding y (starting) values
            min_vals ([type], optional): Highly recommended for global optimization. Defaults to None.
            max_vals ([type], optional): Highly recommended for global optimization. Defaults to None.
        """
        for i in range(len(x)):
            node = self.add_node(x[i], y[i])
            if min_vals is not None:
                node.min_val = min_vals[i]
            if max_vals is not None:
                node.max_val = max_vals[i]


def plot(func, x=np.linspace(0.01, 7.0, 351)):
    y = func(x)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(x, y)
    #These defaults should be fine for most potentials
    ax.set(xlim=[0.0, 7.0], ylim=[-3.0, 3.0], xlabel="r [$\AA$]", ylabel="func(r)")
    return fig, ax
    
