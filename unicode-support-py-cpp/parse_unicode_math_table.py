#!/usr/bin/env python
# -*- coding: utf8 -*-
# :Copyright: © 2011 Günter Milde.

# ===================================================================
# unicode-math-table.tex parser
# ===================================================================
# 
# The file ``unimathsymbols.txt`` contains a mapping between Unicode
# math characters and LaTeX math control sequences. Due to history and
# conceptual differences, this mapping is sometimes ambiguous and
# incomplete.
# 
# This file transforms the unicode-math-table.tex file that comes with 
# the unicode-math package (and is a major source of unimathsymbols.txt)
# into the text format.
# 
# 
# ::

import re, copy

# the unimathsymbols.txt parser/writer module
# 
# ::

from parse_unimathsymbols import *

# Configuration
# -------------
# 
# file names
# ~~~~~~~~~~
# Path to the source file::

urtextname = '/home/milde/texmf/doc/unicode-math/unicode-math-table.tex'



class UrtextEntry(UniMathEntry):

    def __init__(self, line):
        """Parse one `line` of unicode-math-table.tex"""
        self.delimiter = delimiter

# Lines have the form
# 
# ``\UnicodeMathSymbol{"0003A}{\mathcolon      }{\mathpunct}{colon}%``

        line = line.replace(r'\UnicodeMathSymbol{"', '')
        line = line.replace('}%', '')

        fields = [i.strip() for i in line.split('}{')]

# Data fields::

        self.codepoint = int(fields[0], 16) # Unicode Number
        self.utf8 = unichr(self.codepoint).encode('utf8') # literal character in UTF-8 encoding
        self.cmd = ''   # LaTeX command
        self.unicode_math = fields[1]  # macro of the unicode-math package
        self.category = fields[2]      # math category of the symbol
        self.requirements = ''  # package(s) providing the command
        self.comment = fields[3] # aliases and comments



# read_data
# ~~~~~~~~~
# ::

def read_urdata(path=urtextname):
    """Return Table of data entries in the "Urtext" file.
    """
    datafile = file(path, 'r')
    data = Table()

# Read lines and add UniMathEntry instances to the `data` table. 
# Skip comments and empty lines. Use the Unicode character number as key::

    for line in datafile:
        if line.startswith('%') or not line.strip():
            continue
        try:
            entry = UrtextEntry(line)
        except:
            print "error in line", line
            raise
        data[entry.codepoint] = entry

# Close and return::

    datafile.close()
    return data



# Testing
# -------
# ::

if __name__ == '__main__':
    import sys, difflib

    header = read_header()
    urdata = read_urdata()
    
    data = read_data()
    
    urdata.update(data)
    
    data = urdata

    # write_data(urdata, sys.stdout)

# Test for differences after a read-write cycle. Spaces adjacent to the
# delimiter are not significant. ::

    in_lines = file(datafilename, 'r').readlines()
    in_lines = [re.sub(r' *\^ *', '^', line) for line in in_lines]

    header = [re.sub(r' *\^ *', '^', line) for line in header]
    out_lines = [str(v)+'\n' for (k,v) in data]

    diff = ''.join(difflib.unified_diff(in_lines, header + out_lines,
                                        datafilename, '*round trip*'))
    if diff:
        print diff
    else:
        print 'no differences after round trip'

# Write back to outfile::
  
    outfile = None
    # outfile = file('unimathsymbols.test', 'w')
    # outfile = sys.stdout
    if outfile:
        write_data(data, outfile)
        if outfile != sys.stdout:
            print "Output written to", outfile.name


    # for (key, entry) in sort_by_command(data):
    #     print entry

