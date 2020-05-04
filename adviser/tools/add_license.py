import os

def add_license_to_folder(directory: str):
    for subdir, dirs, files in os.walk(directory):
        for f in files:
            ext = os.path.splitext(f)[-1]
            if ext == '.py':
                add_license_to_file(os.path.join(subdir, f))

def add_license_to_file(f_name: str):
    print("- updating license for ", f_name)
    text = []
    add_boiler = False
    with open(f_name, "rt") as in_file:
        for i, line in enumerate(in_file):
            if i == 2:
                if "Copyright 2020" in line:
                    return
                elif "Copyright 2019" in line:
                    line = line.replace("Copyright 2019", "Copyright 2020")
                else:
                    add_boiler = True
            text.append(line)
    if add_boiler:
        text = add_boilerplate(text)
    with open(f_name, "wt") as out_file:
        for line in text:
            out_file.write(line)

def add_boilerplate(text: str):
    boiler = ["############################################################################################\n",
              "#\n",
              "# Copyright 2020, University of Stuttgart: Institute for Natural Language Processing (IMS)\n",
              "#\n",
              "# This file is part of Adviser.\n",
              "# Adviser is free software: you can redistribute it and/or modify'\n",
              "# it under the terms of the GNU General Public License as published by\n",
              "# the Free Software Foundation, either version 3.\n",
              "#\n",
              "# Adviser is distributed in the hope that it will be useful,\n",
              "# but WITHOUT ANY WARRANTY; without even the implied warranty of\n",
              "# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n",
              "# GNU General Public License for more details.\n",
              "#\n",
              "# You should have received a copy of the GNU General Public License\n",
              "# along with Adviser.  If not, see <https://www.gnu.org/licenses/>.\n",
              "#\n",
              "############################################################################################\n",
              "\n"]
    return boiler + text



if __name__ == '__main__':
    directories = ["../examples", "../services", "../tools/regextemplates", "../utils", "../tools/webui"] 
    for d in directories:
        add_license_to_folder(os.path.realpath(d))
    add_license_to_file("../run_chat.py")
    add_license_to_file("../run_demo_multidomain.py")
    add_license_to_file("../tools/getopensmile.py")
    add_license_to_file("../tools/create_ontology.py")

