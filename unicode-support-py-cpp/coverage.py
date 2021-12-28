#!/usr/bin/env python
# -*- coding: utf8 -*-
# :Copyright: © 2011 Günter Milde.
#             Released  without warranties or conditions of any kind
#             under the terms of the Apache License, Version 2.0
#             http://www.apache.org/licenses/LICENSE-2.0
# :Id: $Id:  $

# Find packages providing commands in unimathsymbols.txt
# ================================================================

import sys
import parse_unimathsymbols
from parse_unimathsymbols import superpackages

# Configuration
# -------------

# I/O
outfile = sys.stdout
# outfile = file('../coverage.txt', 'w')

def add_package(package):
    packages[package] = packages.get(package, 0) + 1

header = """\
* %d Unicode math-related symbols 
  (%d without (traditional) LaTeX support)
  
* %d Math-commands provided by LaTeX + packages (including aliases)

* listed command definitions per package (direct and via dependencies):

  .. csv-table:: 
  
"""

if __name__ == '__main__':

    data = parse_unimathsymbols.read_data()
    
    no_of_code_points = len(data)
    no_of_unsupported_symbols = 0
    
    for entry in data.values():
        if not entry.cmd:
            no_of_unsupported_symbols += 1
    
    # parse unimathsymbols for packages
    
    cmds = parse_unimathsymbols.sort_by_command(data)
    packages = parse_unimathsymbols.Table()
    
    for (key, entry) in cmds:
        if not entry.requirements:
            add_package('*standard*')
        for pkg in entry.provided_by():
            add_package(pkg)

    # write data

    # preamble + table header
    outfile.write(header % (no_of_code_points, len(cmds),
                            no_of_unsupported_symbols)
                 )
    
    for key, value in packages:
        outfile.write('   %s, %d\n' %(key, value))

    if outfile != sys.stdout:
        print "Output written to", outfile.name
