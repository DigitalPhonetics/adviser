############################################################################################
#
# Copyright 2020, University of Stuttgart: Institute for Natural Language Processing (IMS)
#
# This file is part of Adviser.
# Adviser is free software: you can redistribute it and/or modify'
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
############################################################################################


import platform
import os


def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_opensmile_executable():
    # check if opensmile binaries exists first

    opensmile_dir = os.path.join(get_root_dir(), "tools", "opensmile-2.3.0")
    os_name = platform.system()
    if os_name == "Darwin" or os_name == "Linux":
        binary_path = os.path.join(opensmile_dir ,"inst", "bin", "SMILExtract")
    elif os_name == "Windows":
        binary_path = os.path.join(opensmile_dir, "bin", "Win32", "SMILExtract_Release.exe")
    #elif os_name == "Linux":
    #    binary_path = os.path.join(opensmile_dir, "bin", "linux_x64_standalone_static", "SMILExtract")

    if os.path.isfile(binary_path):
        # found installation
        return binary_path
    return None
        
def _apply_fix():
    """
    Fixes compiler problem in vectorTransform.hpp, line 117
    Fixes buildscript dependency to -lrt
    Fixes too small ulimit
    """
    
    code = []

    # read in vectorTransform.hpp code
    vectorTransform_hpp = os.path.join(get_root_dir(), "tools", "opensmile-2.3.0", "src", "include", "core", "vectorTransform.hpp")
    with open(vectorTransform_hpp, 'r') as f:
        code = f.readlines()
    # fix line 117
    code[116] = "const unsigned char smileMagic[] = {(unsigned char)0xEE, (unsigned char)0x11, (unsigned char)0x11, (unsigned char)0x00};\n"
    # write fixed code back to file
    with open(vectorTransform_hpp, 'w') as f:
        f.writelines(code)

    # read in buildStandalone.sh code
    buildStandalone_sh = os.path.join(get_root_dir(), "tools", "opensmile-2.3.0", "buildStandalone.sh")
    with open(buildStandalone_sh, 'r') as f:
        code = f.readlines()
    code[6] = "ulimit -n 8000;\n"             # change space to ulimit increase  
    code[47] = 'export LDFLAGS="-lm -lpthread -lc"\n'  # remove -lrt dependency
    with open(buildStandalone_sh, 'w') as f:
        f.writelines(code)


def _download_compile_opensmile():
    from zipfile import ZipFile
    import urllib
    
    opensmile_dir = os.path.join(get_root_dir(), "tools", "opensmile-2.3.0")
    os_name = platform.system()
    opensmile_zip = os.path.join(get_root_dir(), "tools", "opensmile.zip")

    # download zip
    opensmile_url = "https://www.audeering.com/download/opensmile-2-3-0-zip/?wpdmdl=4781"
    print(f"Downloading openSMILE from {opensmile_url} ...")
    urllib.request.urlretrieve(opensmile_url, opensmile_zip)
    # unzip
    print(f"Unzipping to  {opensmile_dir} ...")
    with ZipFile(opensmile_zip, 'r') as zipObj:
        # Extract all the contents of zip file in different directory
        zipObj.extractall(os.path.join(get_root_dir(), 'tools'))
    print("Removing download file ...")
    # delete download
    os.remove(opensmile_zip)

    # compile if neccessary
    if os_name == "Darwin" or os_name == "Linux":
        print("Applying source fixes ...")
        _apply_fix()
        print("Compiling ...")
        import subprocess

        comp_proc = subprocess.Popen(['sh', 'buildStandalone.sh', "p", "$PWD/inst"], stdout=subprocess.PIPE, cwd=opensmile_dir)
        while True:
            output = comp_proc.stdout.readline().strip()
            if output == b'' and comp_proc.poll() is not None:
                break
            if output:
                print(output)
        comp_proc_result = comp_proc.poll()
        print("-------------------------------------------------------")
        print("Compilation process ended with code: ", comp_proc_result)
        print("Marking opensmile library as executable")
        chmodx_proc = subprocess.Popen(['chmod', '+x', "$PWD/inst/bin/SMILExtract"], cwd=opensmile_dir)
        chmodx_result = comp_proc.poll()
        print(chmodx_result)
    # else:  no need to compile - precompiled binaries are shipped with opensmile for win and linux



def get_opensmile_executable_path():
    """
    Returns the path to the platform-specific openSMILE executable.
    If it can't be found, will download (and, depending on platform, try to compile) and then return the path.
    """
    openSmile_path = _get_opensmile_executable()
    if openSmile_path is None:
        print("no openSMILE binary found")
        _download_compile_opensmile()
        openSmile_path = _get_opensmile_executable()
    if openSmile_path is None:
        print("failed to obtain and setup openSMILE. Exiting.")
        exit()
    return openSmile_path
