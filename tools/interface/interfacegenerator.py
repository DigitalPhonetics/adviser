###############################################################################
#
# Copyright 2019, University of Stuttgart: Institute for Natural Language Processing (IMS)
#
# This file is part of Adviser.
# Adviser is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3.
#
# Adviser is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Adviser.  If not, see <https://www.gnu.org/licenses/>.
#
###############################################################################

import importlib.util
import sys
import os
adviser_dir = os.path.realpath("../../")
sys.path.append(adviser_dir)
import inspect
import xml.etree.ElementTree as ET 
from xml.dom import minidom
import glob

from modules.module import Module

# ignore the standard positional arguments from the Module base class
positional_args = [param.name for param in inspect.signature(Module.forward).parameters.values()]


def write_module_interfacedesc_xml(class_name, class_type):
    root = ET.Element('interface')
    inputs = ET.SubElement(root, 'input')
    outputs = ET.SubElement(root, 'output')
    for function in inspect.getmembers(class_type, inspect.isfunction):
        fn_name = function[0]
        if fn_name == 'forward':
            # get signature of forward function
            fn_forward_sig = inspect.signature(class_type.forward)
            return_sig = fn_forward_sig.return_annotation
        
            # filter out unwanted arguments like *kwargs and arguments of Module base class
            keyword_args = [val for val in fn_forward_sig.parameters.values() 
                                if val.kind == val.POSITIONAL_OR_KEYWORD
                                and not val.name in positional_args]
        
            for param in keyword_args:
                # add input arguments
                param_node = ET.SubElement(inputs, 'parameter')
                param_node.set('key', param.name)
                #param_node.set('type', param.annotation.__module__ + '.' + param.annotation.__name__)
                param_node.set('type', str(param.annotation).replace("<class '", '').replace("'>", ''))

            # add output arguments
            if return_sig == inspect._empty:
                print(f"\033[1;31mSignature for function '{fn_name}' of class '{class_name}' is missing a return annotation!\033[0;0m")
            else:
                for output_key in return_sig:
                    output_node = ET.SubElement(outputs, 'parameter')
                    output_node.set('key', output_key)
                    output_node.set('type', str(return_sig[output_key]).replace("<class '", '').replace("'>", ''))
            

    # create interface file
    os.makedirs('interface', exist_ok=True)
    with open(os.path.join('interface', class_name + '.xml'), 'w') as interface_file:
        xmlstr = minidom.parseString(ET.tostring(root, encoding='utf-8')).toprettyxml(indent="   ")
        interface_file.write(xmlstr)


if __name__ == "__main__":
    classes = set()
    # only check modules directory as some of the python files will be executed once they get imported
    for file in glob.glob(os.path.join(adviser_dir, "**", "*.py"), recursive=True):
        # iterate over all python modules
        modname = os.path.splitext(os.path.basename(file))[0]
        if modname != '__init__':
            spec = importlib.util.spec_from_file_location(file, file)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            # mod = __import__(modname)
            classes.update(set(inspect.getmembers(mod, inspect.isclass)))
    classes = {name:value for name, value in classes} # get unique keys

    for class_name, class_type in classes.items():
        # get all classes of module
        if issubclass(class_type, Module) and class_name != 'Module':
            print("Generating interface for Module ", class_name)
            # check if module inherits base Module
            # if True, generate xml interface
            write_module_interfacedesc_xml(class_name, class_type)

