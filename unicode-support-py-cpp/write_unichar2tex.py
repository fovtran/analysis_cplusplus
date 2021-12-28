#!/usr/bin/env python
# -*- coding: utf8 -*-
# :Copyright: © 2011 Günter Milde.
#             Released  without warranties or conditions of any kind
#             under the terms of the Apache License, Version 2.0
#             http://www.apache.org/licenses/LICENSE-2.0

# Write Python source for a Unicode->LaTeX translation table 
# ==========================================================
#
# The translate() method of Python unicode objects expects as `table`
# a dictionary mapping Unicode code points to the expected output.
#
# Usage: (this example works with the "standard" translations and mapping
# of "plain" input to the default math alphabet (``\mathnormal``).
#
# >>> from translation_table import uni2tex_table
# >>> print ur'∫_0^2π sin(x) \d x = 0'.translate(uni2tex_table)
# \int_0^2\pi sin(x) \d x = 0

import sys
import parse_unimathsymbols

# Configuration
# -------------

# Include macros from the following packages::

packages = ['',
            # 'literal', # "feature", no real packge
            'amssymb',
            'amsmath',
            # 'amsxtra',
            # 'bbold',
            # 'esint',
            # 'fourier',
            # 'gensymb',
            # 'isomath',
            # 'kmath',
            # 'lxfonts',
            # 'mathabx',
            # 'mathcomp',
            # 'mathdesign'
            # 'mathdots',
            # 'MnSymbol'
            # 'omlmathit',
            # 'pzccal',
            # 'txfonts',
            # 'stmaryrd',
            # 'wasysym',
            # 'wrisym',
           ]

# codepoint_range = [0, 0x1D800]  # all
codepoint_range =  [0x80, 0x1D800]  # skip ASCII chars
# codepoint_range =  [0, 0x1D400]  # skip Mathematical Alphanumeric Symbols

# I/O
outfile = file('unichar2tex.py', 'w')
# outfile = sys.stdout

# append package name to outfile, if there is only one package
# if len(packages) == 1 and outfile.endswith('.py'):
#     outfile = outfile.replace('.py', packages[0] + '.py')

# Implementation
# ---------------

# preamble + header

header = """\
# LaTeX math to Unicode symbols translation table
# for use with the translate() method of unicode objects.
# Generated with ``write_unichar2tex.py`` from the data in
# http://milde.users.sourceforge.net/LUCR/Math/

# Includes commands from: %s
""" % ', '.join([pkg or 'standard LaTeX' for pkg in packages])

outfile.write(header)

outfile.write("""
uni2tex_table = {
""")

# table lines
data = parse_unimathsymbols.read_data()


for (key, entry) in data:
    if key < codepoint_range[0] or key > codepoint_range[1]:
        continue
    # accents need special handling (swapping argument and accent position)
    if entry.category == 'mathaccent':
        continue
    cmd = entry.supported_cmd(packages)
    if not cmd:
        continue     # skip unsupported chars
    # add space after command name
    if cmd.startswith('\\') and not cmd.endswith('}'):
        cmd += ' '    
    row = "%s: u%r,\n" % (entry.codepoint, cmd)
    outfile.write(row)

outfile.write('}\n')

if outfile != sys.stdout:
    print "Output written to", outfile.name
