#!/usr/bin/env python
# -*- coding: utf8 -*-
# :Copyright: © 2011 Günter Milde.
# :Licence:   This work may be distributed and/or modified under the
#             conditions of the `LaTeX Project Public License`_,
#             either version 1.3 of this license or (at your option)
#             any later version.
# :Id: $Id:  $
#
# ::

"""LaTeX math mode commands and corresponding Unicode characters"""

import sys
import parse_unimathsymbols
from symbols_xetex import *

# Configuration
# -------------

# I/O
outfile = file('../unimathcmds.tex', 'w')
# outfile = sys.stdout

# Include entries for characters without (listed) LaTeX macro? 
# ::

include_unsupported_chars = True
# include_unsupported_chars = False

# Include macros from the following packages::

packages = ['',
            'amssymb', 
            'amsmath',
            # 'amsxtra',
            'bbold',
            # 'esint',
            # 'fourier',
            # 'gensymb',
            'isomath',
            # 'kmath',
            # 'kpfonts',
            # 'lxfonts',
            # 'mathabx',
            # 'mathcomp',
            # 'mathdesign',
            'mathdots',
            # 'MnSymbol',
            # 'oz',
            # 'txfonts',
            'stmaryrd',
            'wasysym',
            # 'wrisym',
           ]


# Implementation
# ---------------

table_head = r"""
\begin{longtable}{llcclll}
\toprule
Command & No. & Text & Math & Category & Requirements & Comments\\
\midrule
\endhead
"""


# Parse and sort data::

data = parse_unimathsymbols.read_data()
data = parse_unimathsymbols.sort_by_command(data)

# preamble + table header
outfile.write(preamble % (package_calls(packages), __doc__))
if [pkg for pkg in packages if pkg and pkg in features]:
    outfile.write(used_features % ', '.join([pkg for pkg in packages 
                                             if pkg and pkg in features]))
outfile.write(used_packages % ', '.join([pkg for pkg in packages 
                                         if pkg not in features]))
outfile.write(table_head)

# table lines
for (key, entry) in data:
    if entry.is_supported(packages):
        # prepare math-macro:
        cmd = entry.cmd
        # add a base character to accent and radical macros:
        if entry.category in ('mathaccent', 'mathradical', 
                              'mathover', 'mathunder'):
            cmd += '{x}'
        if entry.category in ('mathopen'):
            cmd = cmd.replace(r'\lgroup', r'\left\lgroup\right.')
        if entry.category in ('mathclose'):
            cmd = cmd.replace(r'\rgroup', r'\left.\right\rgroup')
        cmd = '$%s$' % cmd
    else:
        if not include_unsupported_chars:
            continue
        cmd = '[na]' # not available with current package selection

    # not supported, if the latex command requires a non-listed package
    if not entry.is_supported(packages):
        math_macro = ''
            
    line = ' & '.join([txt2tex(entry.cmd),
                       '%05X' % entry.codepoint,
                       txt2tex(entry.utf8),
                       cmd, 
                       entry.category, 
                       entry.requirements, 
                       txt2tex(entry.comment),
                      ]) + r' \\'
    outfile.write(line + '\n')

# the end
outfile.write(table_foot)
outfile.write(r'\end{document}')

if outfile != sys.stdout:
    print "%d commands" % len(data)
    print "Output written to ", outfile.name
