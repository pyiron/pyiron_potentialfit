# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
Parsers for MTP/Mlip related files.
"""

import pyparsing as pp
import numpy as np

__author__ = "Marvin Poul"
__copyright__ = "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - " \
                "Computational Materials Design (CM) Department"
__version__ = "1.0"
__maintainer__ = "Marvin Poul"
__email__ = "poul@mpie.de"
__status__ = "development"
__date__ = "Aug 18, 2021"

pp.ParserElement.setDefaultWhitespaceChars(" ")

def _make_potential_parser():
    NL = pp.Suppress("\n")
    EQ = pp.Suppress("=")
    LB = pp.Suppress("{")
    RB = pp.Suppress("}")

    def make_keyword(name):
        return pp.Suppress(pp.Keyword(name))

    def make_field(name, expr, key=None, ungroup=True):
        return pp.ungroup(make_keyword(name) + EQ + expr + NL[0,1]).setResultsName(name if key is None else key)

    def make_list(field_expr, grouped=True):
        list = LB + pp.delimitedList(field_expr) + RB
        if grouped:
            list = pp.Group(list)
        list.setName("list")
        return wrap_numpy(list)

    def wrap_numpy(list_expr):
        list_expr.setParseAction(lambda tokens: np.array(tokens.asList()))
        return list_expr

    indentStack = [1]
    radial_basis_type = make_field("radial_basis_type", "RBChebyshev", "basis_type")
    radial_info = pp.ungroup(pp.indentedBlock(
            make_field("min_dist", pp.pyparsing_common.fnumber) \
          + make_field("max_dist", pp.pyparsing_common.fnumber) \
          + make_field("radial_basis_size", pp.pyparsing_common.integer, "basis_size") \
          + make_field("radial_funcs_count", pp.pyparsing_common.integer, "funcs_count"),
            indentStack
    )).setResultsName("info")

    radial_func_types = pp.Word(pp.nums) + pp.Suppress("-") + pp.Word(pp.nums)
    radial_func_types.setParseAction(lambda tokens: f"{tokens[0]}-{tokens[1]}")
    radial_func_coeffs = LB + pp.delimitedList(pp.pyparsing_common.fnumber) + RB
    radial_funcs = make_keyword("radial_coeffs") + pp.indentedBlock(
                    radial_func_types + pp.indentedBlock(radial_func_coeffs, indentStack)[1, ...],
                indentStack)[1, ...]
    radial_funcs = pp.ungroup(radial_funcs).setResultsName("funcs")
    radial_funcs.setParseAction(lambda tokens: {k: np.array(v) for k, v in tokens.asList()[0]})

    MTP = make_keyword("MTP") + NL

    parser = NL[0, ...] + MTP + pp.Each([
        make_field( "version",        pp.Word(pp.nums + ".") ),
        make_field( "potential_name", pp.Word(pp.alphanums) ),
        make_field( "scaling",        pp.pyparsing_common.fnumber ),
        make_field( "species_count",  pp.pyparsing_common.integer ),
        make_field( "potential_tag",  pp.Optional(pp.Word(pp.alphanums), "")  ),
        pp.Group(radial_basis_type + radial_info + radial_funcs).setResultsName("radial"),
        make_field( "alpha_moments_count", pp.pyparsing_common.integer ),
        make_field( "alpha_index_basic_count", pp.pyparsing_common.integer ),
        make_field( "alpha_index_basic", make_list(make_list(pp.pyparsing_common.integer, grouped=False)) ),
        make_field( "alpha_index_times_count", pp.pyparsing_common.integer ),
        make_field( "alpha_index_times", make_list(make_list(pp.pyparsing_common.integer, grouped=False)) ),
        make_field( "alpha_scalar_moments", pp.pyparsing_common.integer ),
        make_field( "alpha_moment_mapping", make_list(pp.pyparsing_common.integer) ),
        make_field( "species_coeffs", make_list(pp.pyparsing_common.fnumber) ),
        make_field( "moment_coeffs", make_list(pp.pyparsing_common.fnumber) ),
    ])

    return parser

# function exist just to keep namespace clean
_potential_parser = _make_potential_parser()

def potential(potential_string):
    """
    Parse an MTP potential for mlip.

    Args:
        potential_string (str): string to parse

    Raises:
        ValueError: failed to parse potential
    """
    try:
        result = _potential_parser.parseString(potential_string).asDict()
        result['radial']['basis_type'] = result['radial']['basis_type'][2:] # strip RB prefix
        basis_size = result['radial']['info']['basis_size']
        funcs_size = result['radial']['info']['funcs_count']
        for pair, func in result['radial']['funcs'].items():
            if func.shape != (funcs_size, basis_size):
                raise ValueError(f"Invalid radial basis for pair {pair}, should be {funcs_count}x{ basis_size} not {result['radial']['funcs'].shape}")
        if result['alpha_index_basic'].shape != (result['alpha_index_basic_count'], 4):
            raise ValueError(f"Invalid alpha basic indices, length should be {result['alpha_index_basic_count']}")
        if result["alpha_index_times"].shape != (result["alpha_index_times_count"], 4):
            raise ValueError(f"Invalid alpha times indices, length should be {result['alpha_index_times_count']}")
        if len(result["alpha_moment_mapping"]) != result["alpha_scalar_moments"]:
            raise ValueError(f"Invalid alpha moment mapping, length should be {result['alpha_scalar_moments']}")
        if len(result["moment_coeffs"]) != result["alpha_scalar_moments"]:
            raise ValueError(f"Invalid moment coefficients, length should be {result['alpha_scalar_moments']}")
        if len(result["species_coeffs"]) != result["species_count"]:
            raise ValueError(f"Invalid species coefficients, length should be {result['species_count']}")

        return result
    except pp.ParseException:
        raise ValueError("failed to parse potential") from None
